"""Local authorization, durable reservations and budget accounting. No network I/O."""

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


OPERATIONS = {"join", "download", "train", "upload", "submit", "select_final", "read", "worker", "validate"}
BUDGETS = ("money", "gpu_seconds", "wall_seconds", "submissions", "uploads", "train_runs")
COUNTS = {"submissions", "uploads", "train_runs"}
SHA = re.compile(r"[0-9a-f]{64}\Z")


class MandateError(ValueError):
    pass


def encoded(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise MandateError(f"invalid JSON value: {error}") from error


def read_json(path):
    def reject(value):
        raise MandateError(f"non-finite JSON number: {value}")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise MandateError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject, object_pairs_hook=unique)
    except (OSError, json.JSONDecodeError) as error:
        raise MandateError(f"cannot read JSON: {path}: {error}") from error


def timestamp(value=None):
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise MandateError("timestamp must be ISO8601 with timezone") from error
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise MandateError("timestamp must include timezone")
    return value.astimezone(timezone.utc)


def normalize_costs(value, complete=False):
    if not isinstance(value, dict) or set(value) - set(BUDGETS):
        raise MandateError("costs contain unsupported budget dimensions")
    if complete and set(value) != set(BUDGETS):
        raise MandateError("all budget dimensions are required")
    result = dict.fromkeys(BUDGETS, 0)
    for key, amount in value.items():
        if (isinstance(amount, bool) or not isinstance(amount, (int, float))
                or (isinstance(amount, float) and not math.isfinite(amount)) or amount < 0
                or (key in COUNTS and not isinstance(amount, int))):
            raise MandateError(f"invalid nonnegative finite cost: {key}")
        result[key] = amount
    return result


def validate(value):
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise MandateError("unsupported mandate schema")
    if value.get("user_confirmed") is not True:
        raise MandateError("user_confirmed must explicitly be true after user authorization")
    scope = value.get("scope")
    if not isinstance(scope, dict) or not isinstance(scope.get("competition"), str):
        raise MandateError("competition scope is required")
    slug = scope["competition"]
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug):
        raise MandateError("invalid competition slug")
    if scope.get("url") != f"https://www.kaggle.com/competitions/{slug}":
        raise MandateError("competition URL must be canonical HTTPS")
    timestamp(value.get("expires_at", ""))
    operations = value.get("allowed_operations")
    if (not isinstance(operations, list) or not operations
            or any(not isinstance(op, str) or op not in OPERATIONS for op in operations)
            or len(set(operations)) != len(operations)):
        raise MandateError("invalid allowed_operations")
    normalize_costs(value.get("budgets"), complete=True)
    daily = value.get("max_daily_submissions")
    if isinstance(daily, bool) or not isinstance(daily, int) or daily < 0:
        raise MandateError("max_daily_submissions must be a nonnegative integer")
    rules = value.get("rules_acceptance")
    if not isinstance(rules, dict) or not isinstance(rules.get("allowed"), bool):
        raise MandateError("rules_acceptance.allowed must be explicit")
    if rules.get("mode") not in ("pinned", "delegated"):
        raise MandateError("rules_acceptance requires explicit pinned or delegated mode")
    sha = rules.get("document_sha256")
    if sha is not None and (not isinstance(sha, str) or not SHA.fullmatch(sha)):
        raise MandateError("invalid rules document SHA256")
    if rules["mode"] == "pinned" and rules["allowed"] and sha is None:
        raise MandateError("rules acceptance requires the presented document SHA256")
    if rules["mode"] == "delegated":
        constraints = rules.get("constraints")
        if rules["allowed"] is not True or not isinstance(constraints, dict) or any(
                constraints.get(key) is not True for key in
                ("no_payment", "no_publication", "no_personal_disclosure")):
            raise MandateError("delegated rules require explicit authorization and privacy/payment constraints")
    privacy = value.get("privacy")
    if not isinstance(privacy, dict) or any(privacy.get(key) is not True for key in
            ("private_only", "no_publication", "no_personal_disclosure")):
        raise MandateError("privacy restrictions must all be true")
    if value.get("status", "active") not in ("active", "paused", "revoked"):
        raise MandateError("invalid mandate status")
    return copy.deepcopy(value)


def atomic_json(path, value):
    payload = encoded(value) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".mandate-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class MandateStore:
    def __init__(self, root, mandate_file):
        self.root = Path(root).resolve()
        self.mandate_file = Path(mandate_file).resolve()
        self.directory = self.root / ".kaggle"
        self.state_file = self.directory / "mandate-state.json"
        self.audit_file = self.directory / "mandate-audit.jsonl"

    @contextmanager
    def _locked(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "mandate.lock").open("a", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _load(self):
        mandate = validate(read_json(self.mandate_file))
        fingerprint = hashlib.sha256(encoded(mandate).encode()).hexdigest()
        if self.state_file.exists():
            state = read_json(self.state_file)
            if state.get("mandate_sha256") != fingerprint:
                raise MandateError("mandate_changed: cannot self-upgrade or replace an active authorization")
        else:
            state = {"schema_version": 1, "mandate_sha256": fingerprint,
                     "status": mandate.get("status", "active"), "receipts": {}}
        return mandate, state

    def _save(self, state, event, receipt=None):
        entry = {"at": timestamp().isoformat(), "event": event, "receipt": receipt,
                 "status": state["status"], "mandate_sha256": state["mandate_sha256"]}
        payload = encoded(entry) + "\n"
        with self.audit_file.open("a", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(self.state_file, state)

    def _used(self, state):
        used = dict.fromkeys(BUDGETS, 0)
        for receipt in state["receipts"].values():
            charge = receipt.get("actual_costs", receipt["reserved_costs"]) if receipt["status"] == "complete" else receipt["reserved_costs"]
            if receipt["status"] == "overspend":
                charge = {key: max(receipt["reserved_costs"][key], receipt["actual_costs"][key]) for key in BUDGETS}
            for key in BUDGETS:
                used[key] += max(charge[key], receipt["reserved_costs"][key]) if key in COUNTS else charge[key]
        return used

    def reserve(self, operation, competition, identity, costs, metadata=None, now=None):
        if not isinstance(operation, str) or operation not in OPERATIONS:
            raise MandateError("unsupported operation")
        if not isinstance(competition, str):
            raise MandateError("competition must be a slug")
        if not isinstance(identity, (str, dict)) or not identity or (isinstance(identity, str) and not identity.strip()):
            raise MandateError("identity must be a nonempty string or object")
        identity = identity.strip() if isinstance(identity, str) else identity
        metadata = {} if metadata is None else metadata
        if not isinstance(metadata, dict):
            raise MandateError("metadata must be an object")
        encoded(metadata)
        requested = normalize_costs(costs)
        count_key = {"submit": "submissions", "upload": "uploads", "train": "train_runs"}.get(operation)
        if count_key and requested[count_key] != 1:
            raise MandateError(f"{operation} reserves exactly one {count_key}")
        if operation == "read" and any(requested.values()):
            raise MandateError("read operations require zero costs")
        instant = timestamp(now)
        identifier = hashlib.sha256(encoded([competition, operation, identity]).encode()).hexdigest()
        with self._locked():
            mandate, state = self._load()
            if state["status"] != "active":
                raise MandateError(f"mandate is {state['status']}")
            if instant >= timestamp(mandate["expires_at"]):
                raise MandateError("mandate expired")
            if competition != mandate["scope"]["competition"]:
                raise MandateError("competition outside mandate scope")
            if operation not in mandate["allowed_operations"]:
                raise MandateError("operation is not authorized")
            rules = mandate["rules_acceptance"]
            current_rules = metadata.get("rules_document_sha256")
            if operation == "join" and (not rules["allowed"] or current_rules is None):
                raise MandateError("rules acceptance is not authorized for this presented version")
            if current_rules is not None and (not isinstance(current_rules, str) or not SHA.fullmatch(current_rules)):
                raise MandateError("invalid current rules document SHA256")
            if rules["mode"] == "pinned" and current_rules is not None and current_rules != rules["document_sha256"]:
                raise MandateError("rules document changed or is not authorized")
            if operation == "join" and rules["mode"] == "delegated":
                review = metadata.get("rules_review")
                evidence = review.get("evidence") if isinstance(review, dict) else None
                if (not isinstance(review, dict) or review.get("complies") is not True
                        or not isinstance(evidence, (str, dict)) or not evidence
                        or (isinstance(evidence, str) and not evidence.strip())
                        or any(review.get(key) is False for key in
                               ("no_payment", "no_publication", "no_personal_disclosure"))
                        or any(review.get(key) for key in
                               ("payment_required", "publication_required", "personal_disclosure_required"))):
                    raise MandateError("delegated rules require a documented compliant review")
            for key in ("public", "publication", "personal_disclosure"):
                if metadata.get(key):
                    raise MandateError("publication and personal disclosure are prohibited")
            if metadata.get("private") is False:
                raise MandateError("only private operations are authorized")
            if identifier in state["receipts"]:
                previous = state["receipts"][identifier]
                if previous["reserved_costs"] != requested:
                    raise MandateError("duplicate identity has different costs")
                return dict(copy.deepcopy(previous), duplicate=True)
            used = self._used(state)
            for key in BUDGETS:
                if used[key] + requested[key] > mandate["budgets"][key]:
                    raise MandateError(f"budget exhausted: {key}")
            day = instant.date().isoformat()
            daily = sum(item["reserved_costs"]["submissions"] for item in state["receipts"].values() if item["day_utc"] == day)
            if daily + requested["submissions"] > mandate["max_daily_submissions"]:
                raise MandateError("daily submission budget exhausted (UTC)")
            receipt = {"id": identifier, "operation": operation, "competition": competition,
                       "identity": identity, "status": "pending", "reserved_costs": requested,
                       "metadata": metadata, "created_at": instant.isoformat(), "day_utc": day}
            state["receipts"][identifier] = receipt
            self._save(state, "reserve", receipt)
            return dict(copy.deepcopy(receipt), duplicate=False)

    def complete(self, id, result, actual_costs=None):
        if not isinstance(id, str) or not SHA.fullmatch(id):
            raise MandateError("invalid reservation id")
        encoded(result)
        actual = None if actual_costs is None else normalize_costs(actual_costs)
        with self._locked():
            _, state = self._load()
            if id not in state["receipts"]:
                raise MandateError("unknown reservation")
            receipt = state["receipts"][id]
            actual = receipt["reserved_costs"] if actual is None else normalize_costs({**receipt["reserved_costs"], **actual_costs})
            if receipt["status"] in ("complete", "overspend"):
                if receipt["result"] != result or receipt["actual_costs"] != actual:
                    raise MandateError("completion conflicts with recorded result")
                return copy.deepcopy(receipt)
            if receipt["status"] == "failed":
                raise MandateError("known failed reservation cannot be completed")
            overspend = any(actual[key] > receipt["reserved_costs"][key] for key in BUDGETS)
            receipt.update(status="overspend" if overspend else "complete", result=result,
                           actual_costs=actual, completed_at=timestamp().isoformat())
            if overspend:
                state["status"] = "overspend"
            self._save(state, "complete", receipt)
            return copy.deepcopy(receipt)

    def fail(self, id, reason, unknown=True):
        if not isinstance(id, str) or not SHA.fullmatch(id):
            raise MandateError("invalid reservation id")
        if not isinstance(reason, str) or not reason.strip() or not isinstance(unknown, bool):
            raise MandateError("failure requires a reason and boolean unknown")
        with self._locked():
            _, state = self._load()
            if id not in state["receipts"]:
                raise MandateError("unknown reservation")
            receipt = state["receipts"][id]
            status = "unknown" if unknown else "failed"
            if receipt["status"] != "pending" and not (receipt["status"] == "unknown" and unknown is False):
                if receipt["status"] == status and receipt.get("reason") == reason:
                    return copy.deepcopy(receipt)
                raise MandateError("failure conflicts with recorded result")
            receipt.update(status=status, reason=reason, failed_at=timestamp().isoformat())
            self._save(state, "fail", receipt)
            return copy.deepcopy(receipt)

    def inspect(self):
        with self._locked():
            mandate, state = self._load()
            used = self._used(state)
            return {"mandate": mandate, "status": state["status"],
                    "expired": timestamp() >= timestamp(mandate["expires_at"]), "used": used,
                    "remaining": {key: mandate["budgets"][key] - used[key] for key in BUDGETS},
                    "receipts": copy.deepcopy(list(state["receipts"].values()))}

    def revoke(self, reason="revoked by user"):
        return self._stop("revoked", reason)

    def pause(self, reason="paused by user"):
        return self._stop("paused", reason)

    def _stop(self, status, reason):
        if not isinstance(reason, str) or not reason.strip():
            raise MandateError("a reason is required")
        with self._locked():
            _, state = self._load()
            if state["status"] == "revoked" and status != "revoked":
                raise MandateError("revoked mandate cannot be resumed")
            state.update(status=status, stop_reason=reason)
            self._save(state, status)
            return {"status": status, "reason": reason}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--mandate", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "status", "revoke", "pause"):
        commands.add_parser(command)
    reserve = commands.add_parser("reserve")
    for name in ("operation", "competition", "identity", "costs"):
        reserve.add_argument("--" + name, required=True)
    reserve.add_argument("--metadata", default="{}")
    complete = commands.add_parser("complete")
    complete.add_argument("--id", required=True)
    complete.add_argument("--result", required=True)
    complete.add_argument("--actual-costs")
    fail = commands.add_parser("fail")
    fail.add_argument("--id", required=True)
    fail.add_argument("--reason", required=True)
    fail.add_argument("--known", action="store_true")
    args = parser.parse_args()
    store = MandateStore(args.root, args.mandate)
    try:
        if args.command == "validate":
            result = validate(read_json(args.mandate))
        elif args.command == "status":
            result = store.inspect()
        elif args.command in ("revoke", "pause"):
            result = getattr(store, args.command)()
        elif args.command == "reserve":
            result = store.reserve(args.operation, args.competition, json.loads(args.identity),
                                   json.loads(args.costs), json.loads(args.metadata))
        elif args.command == "complete":
            result = store.complete(args.id, json.loads(args.result),
                                    None if args.actual_costs is None else json.loads(args.actual_costs))
        else:
            result = store.fail(args.id, args.reason, unknown=not args.known)
        print(encoded(result))
    except (MandateError, json.JSONDecodeError) as error:
        parser.exit(2, f"mandate error: {error}\n")


if __name__ == "__main__":
    main()
