"""Mandate-bound competition bridge and a persistent, explicitly offline platform."""

from contextlib import contextmanager
import ast
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import shutil
import signal
import stat
import tempfile
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import uuid

from platform_ops import atomic_json, canonical, file_sha
from mandate import normalize_costs


PROTOCOL = "kaggle-competition-bridge/v1"
ACTIONS = ("inspect", "join", "refresh_intel", "download", "upload", "submit", "score",
           "select_final", "final_selection", "reconcile")


class AdapterError(ValueError):
    pass


def _response_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("non-finite JSON number")
    return json.loads(raw, object_pairs_hook=unique, parse_constant=constant)


def _hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _rules(value):
    if not isinstance(value, dict) or not isinstance(value.get("text"), str):
        raise AdapterError("platform did not supply the current rules document")
    actual = hashlib.sha256(value["text"].encode()).hexdigest()
    if value.get("hash", value.get("sha256", actual)) != actual:
        raise AdapterError("rules document hash mismatch")
    if value.get("sha256", actual) != actual or value.get("current") is not True:
        raise AdapterError("rules document is not verified current")
    count = value.get("final_selection_count")
    if type(count) is not int or count < 1:
        raise AdapterError("rules must supply a positive final selection count")
    return {**value, "hash": actual, "sha256": actual}


def _predictions(path):
    if path.suffix.lower() == ".csv":
        rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))
        if len(rows) < 2 or len(rows[0]) < 2 or any(len(row) != len(rows[0]) for row in rows):
            raise AdapterError("submission CSV is empty or not rectangular")
        try:
            values = [float(row[-1]) for row in rows[1:]]
        except ValueError:
            raise AdapterError("submission predictions must be numeric") from None
        if not all(math.isfinite(value) for value in values):
            raise AdapterError("submission predictions must be finite")
        return values
    return None


def validate_candidate(candidate, config=None):
    """Validate the actual bytes; a supplied receipt cannot replace structural checks."""
    config = config or {}
    if not isinstance(candidate, dict) or not isinstance(candidate.get("id"), str) or not candidate["id"]:
        raise AdapterError("candidate requires an id")
    if candidate.get("private") is not True:
        raise AdapterError("candidate upload must be explicitly private")
    path = Path(candidate["path"])
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise AdapterError("candidate must be a nonempty regular file")
    if path.name.lower() in ("kaggle.json", "access_token", ".env"):
        raise AdapterError("credential-like artifact refused")
    actual = file_sha(path)
    if candidate.get("sha256", candidate.get("artifact_sha256")) != actual:
        raise AdapterError("candidate bytes do not match the supplied SHA256")
    kind = candidate.get("kind", "file")
    predictions = None
    if kind == "file":
        predictions = _predictions(path)
    elif kind == "notebook":
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
            raise AdapterError("notebook requires nbformat 4 and cells")
        for cell in notebook["cells"]:
            if not isinstance(cell, dict) or cell.get("cell_type") not in ("code", "markdown", "raw"):
                raise AdapterError("invalid notebook cell")
            source = cell.get("source")
            if not isinstance(source, str) and not (isinstance(source, list) and all(isinstance(s, str) for s in source)):
                raise AdapterError("invalid notebook source")
        if not any(cell["cell_type"] == "code" for cell in notebook["cells"]):
            raise AdapterError("notebook has no code cells")
    elif kind in ("agent", "tar"):
        schema = config.get("agent_schema", {})
        required = schema.get("required_files", ["main.py"])
        entrypoint = schema.get("entrypoint", "main.py")
        maximum = schema.get("max_bytes", 100_000_000)
        if type(maximum) is not int or maximum < 1:
            raise AdapterError("invalid agent archive byte budget")
        with tarfile.open(path, "r:*") as archive:
            members = archive.getmembers()
            names = set()
            total = 0
            for member in members:
                name = PurePosixPath(member.name)
                if (name.is_absolute() or ".." in name.parts or "\\" in member.name
                        or member.issym() or member.islnk() or not (member.isfile() or member.isdir())
                        or member.name in names):
                    raise AdapterError("unsafe or duplicate agent archive member")
                if name.name.lower() in ("kaggle.json", "access_token", ".env"):
                    raise AdapterError("credential-like archive member refused")
                names.add(member.name)
                total += member.size
            if total > maximum or not set(required).issubset(names) or entrypoint not in names:
                raise AdapterError("agent archive does not meet its configured schema")
            if entrypoint.endswith(".py"):
                member = archive.extractfile(entrypoint)
                if member is None:
                    raise AdapterError("agent entrypoint is not a file")
                ast.parse(member.read().decode("utf-8"), filename=entrypoint)
    else:
        raise AdapterError("unsupported candidate kind")
    receipt = candidate.get("validation_receipt")
    if receipt is not None and (not isinstance(receipt, dict) or receipt.get("valid") is not True
            or receipt.get("artifact_sha256") != actual):
        raise AdapterError("validation receipt is stale or failed")
    return {"candidate_id": candidate["id"], "artifact_sha256": actual,
            "private": True, "kind": kind, "path": str(path.resolve()),
            "predictions": predictions, "structural_validation": "passed"}


class CompetitionAdapter:
    def __init__(self, config, mandate_store):
        self.config = dict(config)
        self.store = mandate_store
        self.competition = config.get("competition")
        if not isinstance(self.competition, str) or not self.competition:
            raise AdapterError("competition is required")
        self.backend = config.get("backend")
        if self.backend not in ("fake", "command"):
            raise AdapterError("unknown adapter backend; no live fallback is permitted")
        if self.backend == "fake":
            if not config.get("state_path"):
                raise AdapterError("persistent fake backend requires state_path")
            self.state_path = Path(config["state_path"])
        else:
            command = config.get("adapter_command")
            if command is not None and (not isinstance(command, list) or not command
                    or any(not isinstance(arg, str) or not arg for arg in command)):
                raise AdapterError("adapter_command must be a nonempty argv list")
            timeout = config.get("timeout_seconds", 60)
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
                raise AdapterError("bridge timeout must be positive and finite")
            if command:
                self._bridge_costs()

    def _blocked(self, reason, capability=None):
        return {"status": "blocked", "reason": reason, "capability": capability,
                "competition": self.competition}

    @contextmanager
    def _fake_state(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.state_path.with_suffix(self.state_path.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.state_path.exists():
                state = json.loads(self.state_path.read_text(encoding="utf-8"))
                if state.get("competition") != self.competition:
                    raise AdapterError("fake state belongs to another competition")
            else:
                rules = dict(self.config.get("rules", {"text": "Offline synthetic competition rules", "final_selection_count": 1}))
                rules["current"] = True
                for field in ("payment_required", "publication_required", "personal_disclosure_required"):
                    rules.setdefault(field, False)
                state = {"competition": self.competition, "rules": _rules(rules), "joined": False,
                         "joined_rules_hash": None, "uploads": {}, "submissions": {},
                         "selected": [], "operations": {}, "intel": self.config.get("intel", []),
                         "heldout": self.config.get("heldout", {}), "data": self.config.get("data", {})}
            yield state
            atomic_json(self.state_path, state)

    def _call(self, action, payload=None, operation_id=None):
        if self.backend == "fake":
            return self._fake_call(action, payload or {}, operation_id)
        command = self.config.get("adapter_command")
        if not command:
            return self._blocked("No authenticated command bridge is configured for this platform action", action)
        request = {"protocol": PROTOCOL, "action": action, "competition": self.competition,
                   "payload": payload or {}}
        if operation_id is not None:
            request["operation_id"] = operation_id
        if self.config.get("platform_cli_backend") is not None:
            request["platform_cli_backend"] = self.config["platform_cli_backend"]
        costs = self._bridge_costs()
        mutating = action in ("join", "download", "upload", "submit", "select_final")
        if mutating and not operation_id:
            raise AdapterError("platform mutation requires its reserved operation id")
        identity = {"bridge_protocol": PROTOCOL, "action": action,
                    "platform_operation_id": operation_id if mutating else None,
                    "invocation": None if mutating else uuid.uuid4().hex}
        receipt = self.store.reserve("worker", self.competition, identity, costs,
                                     metadata={"private": True, "bridge_action": action,
                                               "platform_operation_id": operation_id})
        if receipt.get("duplicate"):
            if receipt.get("status") == "complete":
                return receipt["result"]["bridge_result"]
            raise AdapterError("previous bridge invocation is unresolved; reconcile the platform operation")
        try:
            output = self._run_bridge(command, request)
            try:
                envelope = _response_json(output)
            except (ValueError, TypeError):
                raise AdapterError("bridge returned invalid JSON; raw output withheld") from None
            if (not isinstance(envelope, dict) or envelope.get("protocol") != PROTOCOL
                    or envelope.get("action") != action or envelope.get("ok") is not True
                    or not isinstance(envelope.get("result"), dict)):
                raise AdapterError("bridge protocol response failed validation")
        except BaseException:
            self.store.fail(receipt["id"], "bridge invocation did not return a verified response", unknown=True)
            raise
        self.store.complete(receipt["id"], {"bridge_result": envelope["result"]})
        return envelope["result"]

    def _bridge_costs(self):
        value = self.config.get("bridge_max_costs")
        required = {"money", "gpu_seconds", "wall_seconds"}
        if not isinstance(value, dict) or not required.issubset(value):
            raise AdapterError("bridge_max_costs must explicitly declare money, gpu_seconds and wall_seconds")
        costs = normalize_costs(value)
        if (costs["wall_seconds"] < self.config.get("timeout_seconds", 60)
                or any(costs[key] for key in ("submissions", "uploads", "train_runs"))):
            raise AdapterError("bridge wall budget must cover its timeout and operation counts must be zero")
        return costs

    def _run_bridge(self, command, request):
        process = None
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, shell=False, start_new_session=True)
            output, _ = process.communicate(json.dumps(request), timeout=self.config.get("timeout_seconds", 60))
            if process.returncode:
                raise AdapterError("bridge failed; raw process output withheld")
            if len(output) > 8_000_000:
                raise AdapterError("bridge response exceeded its bounded output limit")
            return output
        except BaseException as error:
            if process is not None:
                self._stop_bridge(process)
            if isinstance(error, (subprocess.TimeoutExpired, OSError)):
                raise AdapterError("bridge execution unavailable; mutation outcome may be unknown") from None
            raise
        finally:
            if process is not None:
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()

    def _stop_bridge(self, process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=.5)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()

    def inspect(self):
        result = self._call("inspect")
        if result.get("status") == "blocked":
            return result
        if result.get("competition") != self.competition or not isinstance(result.get("joined"), bool):
            raise AdapterError("platform inspection identity is invalid")
        capabilities = result.get("capabilities")
        if not isinstance(capabilities, dict):
            raise AdapterError("platform inspection requires explicit capabilities")
        return {**result, "rules": _rules(result.get("rules")), "backend": self.backend}

    def _preflight(self, action, require_joined=True):
        info = self.inspect()
        if info.get("status") == "blocked":
            return info
        if info["capabilities"].get(action) is not True:
            return self._blocked("Configured platform bridge does not provide this capability", action)
        if require_joined and action != "join" and (info["joined"] is not True
                or info.get("joined_rules_hash") != info["rules"]["hash"]):
            return self._blocked("Current rules have not been accepted for this competition", "join")
        return info

    def _receipt(self, operation_id):
        return next((item for item in self.store.inspect().get("receipts", []) if item.get("id") == operation_id), None)

    def _normalize(self, action, result, payload, operation_id):
        if result.get("operation_id") != operation_id or result.get("status") != "succeeded":
            raise AdapterError("platform mutation is not confirmed succeeded")
        readback = result.get("readback")
        if not isinstance(readback, dict):
            raise AdapterError("platform mutation lacks authoritative readback")
        if action == "join":
            if readback.get("joined") is not True or readback.get("rules_hash") != payload["rules_hash"]:
                raise AdapterError("rules acceptance readback mismatch")
        elif action == "download":
            files = readback.get("files")
            requested = payload.get("files", [])
            if not isinstance(files, list) or len(files) != len(requested) or not files:
                raise AdapterError("download readback must enumerate actual downloaded files")
            for actual, expected in zip(files, requested):
                if not isinstance(actual, dict) or not isinstance(actual.get("path"), str):
                    raise AdapterError("invalid downloaded file readback")
                path = Path(actual["path"])
                if (path.is_symlink() or not path.is_file() or file_sha(path) != actual.get("sha256")
                        or path.stat().st_size != actual.get("size")
                        or str(path.resolve()) != str(Path(expected["destination"]).resolve())
                        or expected.get("sha256", actual["sha256"]) != actual["sha256"]):
                    raise AdapterError("downloaded file bytes, destination or hash do not match")
        elif action in ("upload", "submit"):
            ref_key = "upload_ref" if action == "upload" else "submission_ref"
            if (not isinstance(readback.get(ref_key), str) or not readback[ref_key]
                    or readback.get("artifact_sha256") != payload["artifact_sha256"]
                    or readback.get("candidate_id") != payload["candidate_id"]):
                raise AdapterError("remote candidate identity readback mismatch")
            if action == "upload" and readback.get("private") is not True:
                raise AdapterError("remote upload privacy was not verified")
        elif action == "select_final":
            if readback.get("refs") != payload["refs"] or readback.get("rules_hash") != payload["rules_hash"]:
                raise AdapterError("final selection readback mismatch")
        return {**readback, "status": "succeeded", "operation_id": operation_id,
                **({"simulation": True} if self.backend == "fake" else {})}

    def _mutate(self, action, payload, costs=None):
        info = self._preflight(action)
        if info.get("status") == "blocked":
            return info
        if payload.get("rules_hash", info["rules"]["hash"]) != info["rules"]["hash"]:
            return self._blocked("Rules changed during the operation preflight", action)
        payload = {**payload, "rules_hash": info["rules"]["hash"]}
        identity_payload = {key: value for key, value in payload.items() if key != "path"}
        metadata = {"adapter_action": action, "adapter_payload": identity_payload,
                    "rules_document_sha256": info["rules"]["hash"], "private": True}
        if action == "join":
            fields = ("payment_required", "publication_required", "personal_disclosure_required")
            if any(info["rules"].get(field) is not False for field in fields):
                return self._blocked("Current rules require or do not rule out payment, publication, or personal disclosure", "join")
            metadata["rules_review"] = {"complies": True, "evidence": {
                "document_sha256": info["rules"]["hash"], **{field: False for field in fields}},
                "no_payment": True, "no_publication": True, "no_personal_disclosure": True}
        receipt = self.store.reserve(action, self.competition,
                                     _hash({"action": action, "payload": identity_payload}), costs or {}, metadata=metadata)
        operation_id = receipt["id"]
        if receipt.get("duplicate"):
            if receipt.get("status") in ("complete", "overspend") and isinstance(receipt.get("result"), dict):
                return receipt["result"]
            return {"status": receipt.get("status", "unknown"), "operation_id": operation_id,
                    "reason": "Existing operation must be reconciled; it will not be executed again"}
        try:
            raw = self._call(action, payload, operation_id)
            result = self._normalize(action, raw, payload, operation_id)
        except (AdapterError, OSError, ValueError, KeyError, TypeError):
            self.store.fail(operation_id, "platform execution or readback not confirmed", unknown=True)
            return {"status": "unknown", "operation_id": operation_id,
                    "reason": "Platform outcome is unknown; reconcile this operation before continuing"}
        self.store.complete(operation_id, result)
        return result

    def join(self, rules_hash):
        info = self.inspect()
        if info.get("status") == "blocked":
            return info
        if rules_hash != info["rules"]["hash"]:
            return self._blocked("Requested rules hash is stale", "join")
        return self._mutate("join", {"rules_hash": rules_hash})

    def refresh_intel(self, cursor=None):
        info = self._preflight("refresh_intel", require_joined=False)
        if info.get("status") == "blocked":
            return info
        result = self._call("refresh_intel", {"cursor": cursor})
        if result.get("status") == "blocked":
            return result
        if not isinstance(result.get("events"), list) or "cursor" not in result:
            raise AdapterError("intel response requires incremental events and a cursor")
        return result

    def download(self, request):
        if not isinstance(request, dict) or not isinstance(request.get("files"), list) or not request["files"]:
            raise AdapterError("download requires a nonempty files list")
        for item in request["files"]:
            if (not isinstance(item, dict) or not isinstance(item.get("destination"), str)
                    or not item["destination"] or not (item.get("source_path") or item.get("url"))):
                raise AdapterError("download requires explicit sources and destinations")
        return self._mutate("download", {"files": request["files"]}, request.get("costs", {}))

    def _snapshot(self, candidate):
        checked = validate_candidate(candidate, self.config)
        base = self.config.get("snapshot_directory")
        if base is None:
            if hasattr(self.store, "root"):
                base = Path(self.store.root) / ".kaggle" / "adapter-artifacts"
            elif self.config.get("state_path"):
                base = Path(self.config["state_path"]).parent / "adapter-artifacts"
            else:
                raise AdapterError("artifact snapshot directory is required")
        folder = Path(base) / checked["artifact_sha256"]
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        if folder.is_symlink():
            raise AdapterError("artifact snapshot directory must not be a symlink")
        target = folder / Path(checked["path"]).name
        if target.exists():
            if target.is_symlink() or not target.is_file() or file_sha(target) != checked["artifact_sha256"]:
                raise AdapterError("immutable artifact snapshot was modified")
        else:
            source_fd = os.open(candidate["path"], os.O_RDONLY | os.O_NOFOLLOW)
            temporary_fd, temporary_name = tempfile.mkstemp(prefix=".snapshot-", dir=folder)
            try:
                with os.fdopen(source_fd, "rb") as source, os.fdopen(temporary_fd, "wb") as destination:
                    if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                        raise AdapterError("snapshot input is not a regular file")
                    shutil.copyfileobj(source, destination)
                    destination.flush()
                    os.fsync(destination.fileno())
                if file_sha(temporary_name) != checked["artifact_sha256"]:
                    raise AdapterError("artifact changed while taking the private snapshot")
                os.chmod(temporary_name, 0o400)
                try:
                    os.link(temporary_name, target)
                except FileExistsError:
                    if target.is_symlink() or file_sha(target) != checked["artifact_sha256"]:
                        raise AdapterError("artifact snapshot collision") from None
            finally:
                Path(temporary_name).unlink(missing_ok=True)
        return validate_candidate({**candidate, "path": str(target)}, self.config)

    def upload(self, candidate):
        payload = self._snapshot(candidate)
        return self._mutate("upload", payload, {**candidate.get("costs", {}), "uploads": 1})

    def submit(self, candidate):
        payload = self._snapshot(candidate)
        if candidate.get("upload_ref"):
            payload["upload_ref"] = candidate["upload_ref"]
        costs = {**candidate.get("costs", {}), "submissions": 1}
        heldout = self.config.get("heldout", {})
        if self.backend == "fake" and heldout.get("prediction_command"):
            timeout = self._prediction_timeout(heldout)
            costs["wall_seconds"] = max(costs.get("wall_seconds", 0), timeout)
            payload["prediction_timeout_seconds"] = timeout
        return self._mutate("submit", payload, costs)

    def score(self, submission):
        info = self._preflight("score")
        if info.get("status") == "blocked":
            return info
        reference = submission if isinstance(submission, str) else submission.get("submission_ref")
        if not isinstance(reference, str) or not reference:
            raise AdapterError("exact submission reference is required")
        known = next((receipt.get("result", {}) for receipt in self.store.inspect().get("receipts", [])
                      if receipt.get("result", {}).get("submission_ref") == reference), None)
        if known is None:
            raise AdapterError("submission reference is not bound to a mandate receipt")
        result = self._call("score", {"submission_ref": reference})
        if result.get("status") in ("pending", "running", "blocked"):
            return result
        value = result.get("score")
        if (result.get("status") != "complete" or result.get("submission_ref") != reference
                or result.get("artifact_sha256") != known.get("artifact_sha256")
                or result.get("candidate_id") != known.get("candidate_id")
                or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise AdapterError("score is not finite and bound to the exact candidate")
        return result

    def select_final(self, refs):
        info = self._preflight("select_final")
        if info.get("status") == "blocked":
            return info
        if (not isinstance(refs, list) or len(refs) != info["rules"]["final_selection_count"]
                or any(not isinstance(ref, str) or not ref for ref in refs) or len(set(refs)) != len(refs)):
            raise AdapterError("final references must match the current rules selection count")
        known = {item.get("result", {}).get("submission_ref") for item in self.store.inspect().get("receipts", [])
                 if item.get("status") in ("complete", "overspend")}
        if not set(refs).issubset(known):
            raise AdapterError("final references must be exact mandate-bound submissions")
        return self._mutate("select_final", {"refs": refs, "rules_hash": info["rules"]["hash"]})

    def final_selection(self):
        info = self._preflight("final_selection", require_joined=False)
        if info.get("status") == "blocked":
            return info
        result = self._call("final_selection")
        if result.get("status") == "blocked":
            return result
        refs = result.get("refs")
        if (not isinstance(refs, list) or any(not isinstance(ref, str) or not ref for ref in refs)
                or len(set(refs)) != len(refs)
                or (refs and len(refs) != info["rules"]["final_selection_count"])
                or result.get("rules_hash") != info["rules"]["hash"]):
            raise AdapterError("invalid final selection readback")
        return result

    def reconcile(self, operation_id):
        receipt = self._receipt(operation_id)
        if receipt is None or receipt.get("competition") != self.competition:
            raise AdapterError("unknown local operation id for this competition")
        if receipt.get("status") in ("complete", "overspend"):
            return receipt["result"]
        if receipt.get("status") not in ("pending", "unknown"):
            return {"status": receipt.get("status"), "operation_id": operation_id}
        metadata = receipt.get("metadata", {})
        action = metadata.get("adapter_action")
        payload = metadata.get("adapter_payload")
        if action not in ACTIONS or not isinstance(payload, dict):
            raise AdapterError("operation is not owned by this adapter")
        try:
            raw = self._call("reconcile", {"operation_id": operation_id}, operation_id)
            if raw.get("status") == "not_found" and raw.get("definitive") is True:
                self.store.fail(operation_id, "bridge proves operation was not executed", unknown=False)
                return {"status": "failed", "operation_id": operation_id}
            result = self._normalize(action, raw, payload, operation_id)
        except (AdapterError, ValueError, KeyError, TypeError):
            return {"status": "unknown", "operation_id": operation_id}
        self.store.complete(operation_id, result)
        return result

    def _prediction_timeout(self, heldout):
        timeout = heldout.get("prediction_timeout_seconds", 10)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise AdapterError("fake prediction timeout must be positive and finite")
        return timeout

    def _fake_predictions(self, state, payload):
        heldout = state["heldout"]
        command = heldout.get("prediction_command")
        if command is None:
            return _predictions(Path(payload["path"])) if payload["kind"] == "file" else None
        if (not isinstance(command, list) or not command
                or any(not isinstance(arg, str) or not arg for arg in command)):
            raise AdapterError("fake prediction_command requires a trusted explicit argv list")
        timeout = self._prediction_timeout(heldout)
        if payload.get("prediction_timeout_seconds", 0) < timeout:
            raise AdapterError("fake inference does not have a reserved wall-time budget")
        inputs = heldout.get("inputs")
        if not isinstance(inputs, list):
            raise AdapterError("fake predictor requires synthetic hidden inputs")
        request = {"artifact_path": payload["path"], "kind": payload["kind"], "inputs": inputs}
        try:
            response = subprocess.run(command, input=json.dumps(request), capture_output=True, text=True,
                                      shell=False, timeout=timeout)
        except (subprocess.TimeoutExpired, OSError):
            raise AdapterError("fake prediction driver failed or exceeded reserved time") from None
        if response.returncode or len(response.stdout) > 8_000_000:
            raise AdapterError("fake prediction driver did not produce a bounded successful result")
        result = _response_json(response.stdout)
        predictions = result.get("predictions") if isinstance(result, dict) else None
        if (not isinstance(predictions, list) or len(predictions) != len(inputs)
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in predictions)
                or file_sha(payload["path"]) != payload["artifact_sha256"]):
            raise AdapterError("fake prediction result or artifact identity is invalid")
        return predictions

    def _fake_call(self, action, payload, operation_id):
        with self._fake_state() as state:
            if action == "inspect":
                return {"competition": self.competition, "capabilities": {key: True for key in ACTIONS},
                        "rules": _rules(state["rules"]), "joined": state["joined"],
                        "joined_rules_hash": state["joined_rules_hash"], "data": state.get("data", {}), "simulation": True}
            if action == "refresh_intel":
                cursor = payload.get("cursor")
                start = 0 if cursor is None else cursor
                if type(start) is not int or not 0 <= start <= len(state["intel"]):
                    raise AdapterError("invalid fake intelligence cursor")
                return {"cursor": len(state["intel"]), "events": state["intel"][start:], "simulation": True}
            if action == "reconcile":
                return state["operations"].get(payload["operation_id"],
                    {"status": "not_found", "definitive": True, "operation_id": operation_id})
            if action == "final_selection":
                return {"refs": state["selected"], "rules_hash": state["rules"]["hash"], "simulation": True}
            if action == "score":
                item = state["submissions"].get(payload["submission_ref"])
                if item is None:
                    raise AdapterError("fake submission does not exist")
                heldout = state["heldout"]
                if item["artifact_sha256"] in heldout.get("scores_by_sha256", {}):
                    value = heldout["scores_by_sha256"][item["artifact_sha256"]]
                else:
                    labels, predictions = heldout.get("labels"), item.get("predictions")
                    if (not isinstance(labels, list) or not labels or not isinstance(predictions, list)
                            or len(labels) != len(predictions)):
                        raise AdapterError("fake heldout evaluator requires aligned labels and predictions")
                    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                           for v in labels + predictions):
                        raise AdapterError("fake heldout values must be finite numbers")
                    metric = heldout.get("metric", "accuracy")
                    if metric == "accuracy":
                        value = sum((pred >= .5) == bool(label) for pred, label in zip(predictions, labels)) / len(labels)
                    elif metric in ("mse", "rmse", "mae"):
                        errors = [(abs(a-b) if metric == "mae" else (a-b)**2) for a, b in zip(predictions, labels)]
                        value = sum(errors) / len(errors)
                        if metric == "rmse":
                            value = math.sqrt(value)
                    else:
                        raise AdapterError("unsupported fake evaluator metric")
                return {"submission_ref": payload["submission_ref"], "candidate_id": item["candidate_id"],
                        "artifact_sha256": item["artifact_sha256"], "status": "complete", "score": value,
                        "simulation": True, "metric": heldout.get("metric", "configured")}
            if operation_id in state["operations"]:
                return state["operations"][operation_id]
            if action == "join":
                if payload["rules_hash"] != _rules(state["rules"])["hash"]:
                    raise AdapterError("fake rules changed")
                state["joined"], state["joined_rules_hash"] = True, payload["rules_hash"]
                readback = {"joined": True, "rules_hash": payload["rules_hash"]}
            elif action == "download":
                allowed = {str(Path(value).resolve()) for key, value in state.get("data", {}).items()
                           if key.endswith("_path") and isinstance(value, str)}
                files = []
                for item in payload["files"]:
                    source = Path(item.get("source_path", ""))
                    destination = Path(item["destination"])
                    if source.is_symlink() or not source.is_file() or str(source.resolve()) not in allowed:
                        raise AdapterError("fake download source is not an explicitly configured public data file")
                    expected = file_sha(source)
                    if item.get("sha256", expected) != expected:
                        raise AdapterError("fake download source hash mismatch")
                    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    if destination.exists():
                        if destination.is_symlink() or not destination.is_file() or file_sha(destination) != expected:
                            raise AdapterError("fake download refuses to overwrite different existing bytes")
                    else:
                        with source.open("rb") as input_file, destination.open("xb") as output_file:
                            os.fchmod(output_file.fileno(), 0o600)
                            shutil.copyfileobj(input_file, output_file)
                    files.append({"path": str(destination.resolve()), "sha256": file_sha(destination),
                                  "size": destination.stat().st_size})
                readback = {"files": files}
            elif action in ("upload", "submit"):
                if not state["joined"] or state["joined_rules_hash"] != _rules(state["rules"])["hash"]:
                    raise AdapterError("current fake rules must be joined first")
                if file_sha(payload["path"]) != payload["artifact_sha256"] or payload["private"] is not True:
                    raise AdapterError("fake artifact changed before platform accepted it")
                key = "uploads" if action == "upload" else "submissions"
                reference = ("upload-" if action == "upload" else "submission-") + operation_id
                if action == "submit" and payload.get("upload_ref"):
                    uploaded = state["uploads"].get(payload["upload_ref"])
                    if not uploaded or uploaded["artifact_sha256"] != payload["artifact_sha256"]:
                        raise AdapterError("uploaded artifact identity mismatch")
                predictions = self._fake_predictions(state, payload) if action == "submit" else None
                state[key][reference] = {**payload, "predictions": predictions}
                readback = {("upload_ref" if action == "upload" else "submission_ref"): reference,
                            "candidate_id": payload["candidate_id"], "artifact_sha256": payload["artifact_sha256"],
                            "private": True}
            elif action == "select_final":
                rules = _rules(state["rules"])
                if (len(payload["refs"]) != rules["final_selection_count"]
                        or not set(payload["refs"]).issubset(state["submissions"])):
                    raise AdapterError("fake final references violate the rules")
                state["selected"] = payload["refs"]
                readback = {"refs": payload["refs"], "rules_hash": rules["hash"]}
            else:
                raise AdapterError("unsupported fake platform action")
            result = {"operation_id": operation_id, "status": "succeeded", "readback": readback, "simulation": True}
            state["operations"][operation_id] = result
            return result
