#!/usr/bin/env python3
"""Persistent, budget-reserved AI worker orchestration; standard library only."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import analysis
import state_store
import workspace
from mandate import MandateStore


class ManagedError(ValueError):
    pass


def instant(value=None):
    result = datetime.now(timezone.utc) if value is None else value
    if isinstance(result, str):
        result = datetime.fromisoformat(result.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ManagedError("time requires timezone")
    return result


def strict_json(value):
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ManagedError("duplicate worker JSON key: " + key)
            result[key] = item
        return result
    def reject(value):
        raise ManagedError("non-finite JSON: " + value)
    result = json.loads(value, object_pairs_hook=unique, parse_constant=reject)
    if not isinstance(result, dict):
        raise ManagedError("worker must return one JSON object")
    json.dumps(result, allow_nan=False)
    return result


def command(argv, root, payload, timeout, new_session=True):
    if not isinstance(argv, list) or not argv or any(not isinstance(v, str) or not v for v in argv):
        raise ManagedError("command must be a nonempty argv list; shell strings are forbidden")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ManagedError("positive command timeout required")
    process = subprocess.Popen(argv, cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=new_session)
    try:
        stdout, stderr = process.communicate(json.dumps(payload, allow_nan=False), timeout=timeout)
    except BaseException:
        if new_session:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        if new_session:
            process.communicate()
        else:
            process.wait()
            process.stdout.close()
            process.stderr.close()
        raise
    if process.returncode:
        # Child logs may contain credentials; never include them in receipts.
        raise ManagedError(f"command failed with exit status {process.returncode}")
    if len(stdout) > 8_000_000:
        raise ManagedError("command JSON response exceeds 8MB")
    return strict_json(stdout) if stdout.strip() else {}


def supervised_job(request_path):
    """Runs independently of the scheduler and leaves an atomic completion receipt."""
    request = state_store.read_json(request_path)
    result_path = Path(request["result_path"])
    if result_path.exists():
        raise ManagedError("local job already has a result")
    state_store.atomic_json(Path(request["process_path"]), {"pid": os.getpid(), "operation_id": request["operation_id"],
                                                          "started_at": instant().isoformat()})
    try:
        result = command(request["argv"], Path(request["root"]), request["payload"], request["timeout"], new_session=False)
    except BaseException as error:
        state_store.atomic_json(result_path, {"operation_id": request["operation_id"], "status": "unknown",
                                             "error_type": type(error).__name__, "finished_at": instant().isoformat()})
        if os.getpgrp() == os.getpid():
            os.killpg(os.getpid(), signal.SIGKILL)
        raise
    state_store.atomic_json(result_path, {"operation_id": request["operation_id"], "status": "complete",
                                         "result": result, "finished_at": instant().isoformat()})
    return result


class ManagedRunner:
    def __init__(self, config, adapter=None, store=None):
        self.config = dict(config)
        self.user_goal = config.get("goal")
        self.root = Path(config["root"]).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        agent = str(Path(__file__).resolve().with_name("agent_process.py"))
        default_timeout = config.get("worker_timeout_seconds", 300)
        self.config.setdefault("worker_command", [sys.executable, agent, "--role", "worker", "--workspace", str(self.root),
                                                  "--model", config.get("model", "gpt-6.1-sol"), "--timeout", str(max(1, default_timeout - 5))])
        self.path = self.root / "managed-state.json"
        self.store = store or MandateStore(self.root, config["mandate_file"])
        if adapter is None:
            from competition_adapter import CompetitionAdapter
            adapter_config = dict(config.get("adapter", {"backend": "command"}))
            adapter_config.setdefault("competition", config["competition"])
            adapter_config.setdefault("url", self.store.inspect()["mandate"].get("scope", {}).get("url"))
            if adapter_config.get("backend") == "command":
                bridge_timeout = adapter_config.get("timeout_seconds", 300)
                adapter_config.setdefault("adapter_command", [sys.executable, agent, "--role", "platform", "--workspace", str(self.root),
                                                              "--model", config.get("model", "gpt-6.1-sol"), "--timeout", str(max(1, bridge_timeout - 5))])
                adapter_config.setdefault("timeout_seconds", bridge_timeout)
                bridge_costs = dict(config.get("worker_max_costs", {}))
                bridge_costs.setdefault("gpu_seconds", 0)
                bridge_costs["wall_seconds"] = max(bridge_timeout, bridge_costs.get("wall_seconds", 0))
                adapter_config.setdefault("bridge_max_costs", bridge_costs)
            adapter = CompetitionAdapter(adapter_config, self.store)
        self.adapter = adapter
        self.state = self._read()
        self.config.update(self.state.get("setup", {}))
        if self.user_goal is not None:
            self.config["goal"] = self.user_goal

    def _read(self):
        if self.path.exists():
            return state_store.read_json(self.path)
        return {"schema": 1, "status": "ready", "phase": "bootstrap", "attempt": 0,
                "next_due_at": None, "intel_due_at": None, "intel_cursor": None,
                "intel": {}, "queue": [], "jobs": [], "no_gain": 0, "errors": 0,
                "frozen": False, "best_score": None, "history": []}

    def _save(self):
        state_store.atomic_json(self.path, self.state)

    @contextmanager
    def _lock(self):
        with (self.root / ".managed.lock").open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ManagedError("another managed runner owns this workspace") from error
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _path(self, value):
        path = Path(value)
        path = path if path.is_absolute() else self.root / path
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ManagedError("artifact path escapes workspace")
        return resolved

    def _files(self, files):
        for item in files:
            if not isinstance(item, dict) or set(item) != {"path", "content"} or not isinstance(item["content"], str):
                raise ManagedError("files require exactly path and content")
            path = self._path(item["path"])
            if path.exists():
                if path.read_text() == item["content"]:
                    continue
                raise ManagedError("worker cannot overwrite existing files: " + item["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                stream.write(item["content"])

    def _budgeted(self, operation, identity, costs, argv, payload, timeout):
        costs = dict(costs)
        if operation == "train":
            costs["train_runs"] = 1
        costs.setdefault("wall_seconds", timeout)
        if costs["wall_seconds"] < timeout:
            raise ManagedError("wall_seconds reservation must cover command timeout")
        for key, limit in self.config.get("command_max_costs", {}).items():
            if costs.get(key, 0) > limit:
                raise ManagedError("process exceeds configured per-command budget: " + key)
        reservation = self.store.reserve(operation, self.config["competition"], identity, costs,
                                         metadata={"private": True})
        if reservation.get("duplicate"):
            self._recover_jobs()
            reservation = next(item for item in self.store.inspect()["receipts"] if item["id"] == reservation["id"])
            if reservation["status"] == "complete":
                return reservation["result"]
            raise ManagedError("unresolved local operation; refusing duplicate process")
        folder = self.root / ".kaggle" / "jobs"
        folder.mkdir(parents=True, exist_ok=True)
        result_path = folder / (reservation["id"] + ".result.json")
        request_path = folder / (reservation["id"] + ".request.json")
        process_path = folder / (reservation["id"] + ".process.json")
        state_store.atomic_json(request_path, {"operation_id": reservation["id"], "argv": argv, "root": str(self.root),
                                              "payload": payload, "timeout": timeout, "result_path": str(result_path),
                                              "process_path": str(process_path)})
        self.state["jobs"].append({"id": reservation["id"], "status": "running", "identity": identity,
                                    "result_path": str(result_path), "process_path": str(process_path)})
        self._save()
        try:
            wrapper = [sys.executable, str(Path(__file__).resolve()), "_job", str(request_path)]
            result = command(wrapper, self.root, {}, timeout)
        except BaseException as error:
            self.store.fail(reservation["id"], type(error).__name__, unknown=True)
            self.state["jobs"][-1]["status"] = "unknown"
            self._save()
            raise
        self.store.complete(reservation["id"], result)
        self.state["jobs"][-1]["status"] = "complete"
        self._save()
        return result

    def _recover_jobs(self):
        receipts = {item["id"]: item for item in self.store.inspect()["receipts"]}
        for job in self.state["jobs"]:
            receipt = receipts.get(job["id"])
            if receipt is None or receipt["status"] == "complete" or not job.get("result_path"):
                continue
            path = self._path(job["result_path"])
            if not path.exists():
                continue
            result = state_store.read_json(path)
            if result.get("operation_id") != job["id"]:
                raise ManagedError("local supervisor operation identity differs")
            if result.get("status") == "complete":
                self.store.complete(job["id"], result["result"])
                job["status"] = "complete"
        self._save()

    def _facts(self):
        if not (self.root / "competition.json").exists():
            return {"competition": None, "ledger": None}
        config, ledger = state_store.load(self.root)
        return {"competition": config, "ledger": ledger}

    def _worker(self, kind, now, **extra):
        timeout = self.config.get("worker_timeout_seconds", 300)
        if "money" not in self.config.get("worker_max_costs", {}):
            raise ManagedError("worker_max_costs.money must explicitly bound AI cost; subscription/local calls may set zero")
        budget = self.store.inspect()
        scope = budget["mandate"].get("scope", {})
        url = scope.get("url", self.config.get("url", self.config.get("competition_url")))
        request = {"schema": 1, "task_kind": kind, "workspace": str(self.root),
                   "skill_path": str(Path(__file__).resolve().parents[1] / "SKILL.md"),
                   "goal": self.config.get("goal"), "objective": self.config.get("goal"),
                   "scope": scope, "competition_url": url,
                   "facts": self._facts(), "platform": {**(self.state.get("platform") or {}), "url": url},
                   "intel": self.state["intel"], "budget": budget,
                   "route": self._route(now), "attempt": self.state["attempt"],
                   "remaining_seconds": self._remaining(now), **extra}
        result = self._budgeted("worker", {"task": kind, "attempt": self.state["attempt"],
                                          "cursor": self.state["intel_cursor"]},
                                self.config.get("worker_max_costs", {}), self.config["worker_command"],
                                request, timeout)
        self._schema(result)
        return result

    @staticmethod
    def _schema(result):
        common = {"kind", "reason"}
        allowed = {
            "bootstrap": common | {"files", "command", "costs", "setup", "downloads"},
            "experiment": common | {"registration", "files", "command", "costs", "outputs"},
            "holdout": common | {"files", "command", "costs", "outputs"},
            "stop": common, "wait": common | {"seconds"},
        }
        if result.get("kind") not in allowed or set(result) - allowed[result["kind"]]:
            raise ManagedError("unknown worker response kind or fields")
        if not isinstance(result.get("reason"), str) or not result["reason"].strip():
            raise ManagedError("worker decision requires a reason")

    def _remaining(self, now):
        deadline = self.config.get("deadline") or self._facts()["competition"] and self._facts()["competition"].get("deadline")
        return None if deadline is None else (instant(deadline) - now).total_seconds()

    def _route(self, now):
        remaining = self._remaining(now)
        budget = self.store.inspect()["remaining"]
        if self.state["frozen"] or remaining is not None and remaining <= self.config.get("freeze_hours", 24) * 3600:
            return "freeze_and_finalize"
        if remaining is not None and remaining < 7 * 86400:
            return "minimal_valid_baseline_then_low_compute"
        if budget.get("gpu_seconds", 0) <= 0:
            return "cpu_low_compute"
        return "structural_experiments"

    def _execute(self, plan, task):
        self._files(plan.get("files", []))
        timeout = self.config.get("training_max_seconds", 600)
        ceiling = self.config.get("command_max_costs", {})
        costs = plan.get("costs", {})
        if any(key not in costs for key in ("money", "gpu_seconds")):
            raise ManagedError("execution plan must explicitly bound money and GPU seconds")
        for key, amount in costs.items():
            if isinstance(amount, bool) or not isinstance(amount, (float, int)) or not math.isfinite(amount) or amount < 0:
                raise ManagedError("invalid command cost")
            if key in ceiling and amount > ceiling[key]:
                raise ManagedError("command exceeds configured per-command budget")
        return self._budgeted("train" if task != "holdout" else "validate",
                              {"task": task, "attempt": self.state["attempt"]}, costs,
                              plan["command"], {"task_kind": task, "workspace": str(self.root)}, timeout)

    def _rows(self, path, truth_path):
        rows = analysis.read_rows(path)
        truth = analysis.csv_records(truth_path, ["row_id", "unit", "fold", "y_true"])
        by_id = {row["row_id"]: row for row in truth}
        if len(by_id) != len(truth) or set(by_id) != {row["row_id"] for row in rows}:
            raise ManagedError("prediction row identities differ from sealed truth")
        for row in rows:
            expected = by_id[row["row_id"]]
            if any(row.get(key) != expected.get(key) for key in ("unit", "fold", "group")) or row["y_true"] != float(expected["y_true"]):
                raise ManagedError("prediction truth, fold, or group differs from registered data")
        return rows

    def _preflight(self):
        required = ("validation_truth", "baseline_rows", "sample_submission", "holdout_ids", "holdout_truth", "validator_command")
        if any(not self.config.get(key) for key in required):
            raise ManagedError("bootstrap must prepare truth, baseline, sample, sealed holdout and independent validator")
        rows = self._rows(self._path(self.config["baseline_rows"]), self._path(self.config["validation_truth"]))
        held = analysis.csv_records(self._path(self.config["holdout_truth"]), ["row_id", "unit", "fold", "y_true"])
        ids = state_store.read_json(self._path(self.config["holdout_ids"]))
        if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)) or set(ids) != {r["unit"] for r in held}:
            raise ManagedError("sealed holdout ID manifest differs from holdout truth units")
        if {r["unit"] for r in rows} & set(ids) or {r["row_id"] for r in rows} & {r["row_id"] for r in held}:
            raise ManagedError("development and sealed holdout overlap")
        if {r.get("group") for r in rows} - {None, ""} & {r.get("group") for r in held} - {None, ""}:
            raise ManagedError("development and sealed holdout groups overlap")
        if len({r["fold"] for r in rows}) < self.config.get("gate", {}).get("min_folds", 3):
            raise ManagedError("insufficient development folds")
        sample = analysis.csv_records(self._path(self.config["sample_submission"]),
                                      [self.config.get("id_column", "id"), self.config.get("prediction_column", "prediction")])
        id_column = self.config.get("id_column", "id")
        if len({r[id_column] for r in sample}) != len(sample):
            raise ManagedError("duplicate sample submission ID")
        return {"decision": "PASS", "reason": "verified exact truth identity, separated holdout units, folds and unique submission IDs"}

    def _bootstrap(self, now):
        if all(self.config.get(key) for key in ("validation_truth", "baseline_rows", "holdout_ids", "holdout_truth", "sample_submission", "validator_command")) and (self.root / "competition.json").exists():
            self._preflight()
        else:
            plan = self.state.get("bootstrap_plan") or self._worker("bootstrap", now)
            if plan["kind"] != "bootstrap":
                raise ManagedError("first worker task must prepare a bootstrap")
            self.state["bootstrap_plan"] = plan
            self._save()
            downloads = self.state.setdefault("bootstrap_downloads", {})
            for index, request in enumerate(plan.get("downloads", [])):
                key = str(index)
                if downloads.get(key, {}).get("status") in ("succeeded", "complete"):
                    continue
                request = dict(request)
                request["files"] = [{**item, "destination": str(self._path(item["destination"]))} for item in request["files"]]
                if downloads.get(key, {}).get("operation_id"):
                    downloaded = self.adapter.reconcile(downloads[key]["operation_id"])
                else:
                    downloaded = self.adapter.download(request)
                downloads[key] = downloaded
                self._save()
                if downloaded.get("status") in ("unknown", "pending", "queued"):
                    return
                if downloaded.get("status") not in ("succeeded", "complete"):
                    raise ManagedError("authorized dataset download not confirmed")
            self._execute(plan, "bootstrap")
            setup = plan["setup"]
            allowed = {"goal", "metric", "unit_metric", "deadline", "reserve_hours", "route", "validation_truth", "baseline_rows", "sample_submission", "holdout_ids", "holdout_truth", "validator_command", "id_column", "prediction_column", "training_data", "scorer"}
            if not isinstance(setup, dict) or set(setup) - allowed:
                raise ManagedError("invalid bootstrap setup")
            if self.user_goal is not None:
                setup["goal"] = self.user_goal
            self.config.update(setup)
            if not state_store.metric_valid(setup.get("metric")) or not setup.get("deadline"):
                raise ManagedError("bootstrap must discover official metric and deadline")
            secondary = {"auc": "logloss", "qwk": "mae"}.get(setup["metric"]["name"])
            if secondary and not self.config.get("unit_metric"):
                setup["unit_metric"] = secondary
                self.config["unit_metric"] = secondary
                self.state["history"].append({"event": "secondary_unit_metric", "primary": setup["metric"]["name"],
                                              "unit_metric": secondary, "reason": "primary may be undefined on single-class units; pooled primary metric retains promotion authority", "at": now.isoformat()})
            official = self.state.get("platform", {})
            metric = official.get("metric", official.get("rules", {}).get("metric"))
            deadline = official.get("deadline", official.get("rules", {}).get("deadline"))
            if metric is not None and setup["metric"] != metric:
                raise ManagedError("worker metric differs from official platform metric")
            if deadline is not None and instant(setup["deadline"]) != instant(deadline):
                raise ManagedError("worker deadline differs from official platform deadline")
            self._preflight()
            if not (self.root / "competition.json").exists():
                workspace.initialize(self.root)
            state_store.transact(self.root, "config", {"competition": self.config["competition"],
                                 "metric": setup["metric"], "deadline": setup["deadline"],
                                 "goal": self.config.get("goal"),
                                 "reserve_hours": setup.get("reserve_hours", self.config.get("freeze_hours", 24)),
                                 "route": setup.get("route", self._route(now)), "started_at": now.isoformat()})
            self.state["setup"] = setup
        _, ledger = state_store.load(self.root)
        if not ledger["holdout"].get("sha256"):
            state_store.transact(self.root, "holdout-register", artifact=self._path(self.config["holdout_ids"]))
        self.state["phase"] = "iterate"
        self.state.pop("bootstrap_plan", None)
        self._save()

    def _artifact(self, artifact, outputs, identifier):
        from competition_adapter import validate_candidate
        kind = outputs.get("kind", "file")
        validate_candidate({"id": identifier, "path": str(artifact), "sha256": state_store.digest(artifact),
                            "private": True, "kind": kind}, self.config.get("adapter", {}))
        if kind != "file" and not outputs.get("predictions"):
            raise ManagedError("notebook/agent output requires independently verifiable prediction CSV")
        prediction_path = self._path(outputs["predictions"]) if outputs.get("predictions") else artifact
        id_column = self.config.get("id_column", "id")
        pred_column = self.config.get("prediction_column", "prediction")
        sample = analysis.csv_records(self._path(self.config["sample_submission"]), [id_column, pred_column])
        rows = analysis.csv_records(prediction_path, [id_column, pred_column])
        if [r[id_column] for r in rows] != [r[id_column] for r in sample]:
            raise ManagedError("submission ID order or count differs from sample")
        for row in rows:
            analysis.finite(row[pred_column], "submission prediction")

    def _score_rows(self, path):
        metric = state_store.load(self.root)[0]["metric"]["name"]
        argv = ["score", "--rows", str(path), "--metric", metric]
        if self.config.get("unit_metric"):
            argv += ["--unit-metric", self.config["unit_metric"]]
        if self.config.get("scorer"):
            argv += ["--scorer", self._scorer()]
        args = analysis.parser().parse_args(argv)
        return args.run(args)

    def _scorer(self):
        path, separator, function = self.config["scorer"].rpartition(":")
        if not separator or not function:
            raise ManagedError("custom scorer requires path.py:function")
        return str(self._path(path)) + ":" + function

    def _units(self, rows_path, output_path):
        result = self._score_rows(rows_path)
        with output_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["unit", "fold", "group", "score"])
            writer.writeheader()
            for unit in result["units"]:
                writer.writerow({key: unit[key] for key in writer.fieldnames})
        return result

    def _validator(self, item, artifact, rows, now, holdout=False, predictions=None, kind="file"):
        timeout = self.config.get("validator_timeout_seconds", 120)
        receipt = self._budgeted("validate", {"candidate": item["id"], "sha256": state_store.digest(artifact), "rows_sha256": state_store.digest(rows), "kind": "holdout" if holdout else "pipeline"},
                                 self.config.get("validator_max_costs", {}), self.config["validator_command"],
                                 {"task_kind": "holdout" if holdout else "pipeline", "candidate": item,
                                  "outputs": {"artifact": str(artifact), "rows": str(rows), "kind": kind,
                                              "predictions": str(predictions) if predictions else str(artifact)},
                                  "truth_path": str(self._path(self.config["holdout_truth" if holdout else "validation_truth"])),
                                  "sample_submission": str(self._path(self.config["sample_submission"])),
                                  "baseline_rows": str(self._path(self.config["baseline_rows"])),
                                  "stages": item["stages"], "metric": state_store.load(self.root)[0]["metric"]}, timeout)
        checks = receipt.get("checks", {})
        if (receipt.get("decision") != "PASS" or receipt.get("artifact_sha256") != state_store.digest(artifact)
                or receipt.get("rows_sha256") != state_store.digest(rows)
                or not set(analysis.STAGES) <= set(receipt.get("covered_stages", []))
                or any(checks.get(k) is not True for k in ("prediction_to_artifact", "split_isolation", "training_provenance"))):
            raise ManagedError("independent validator did not verify pipeline, split, provenance and artifact hashes")
        if holdout:
            return receipt
        state_store.transact(self.root, "validator", {"authority": "full", "covered_stages": receipt["covered_stages"], "receipt": receipt})
        preflight = self._preflight()
        state_store.transact(self.root, "config", {"gates": {key: preflight for key in ("strategy", "test_distribution", "validation")}})

    def _candidate(self, plan, now):
        item = plan["registration"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", item.get("id", "")):
            raise ManagedError("candidate ID must be a safe filename component")
        if set(item.get("stages", [])) != set(analysis.STAGES):
            raise ManagedError("managed candidates must register the complete pipeline")
        state_store.transact(self.root, "register", item)
        self.state["active_plan"] = plan
        self._save()
        self._execute(plan, "experiment")
        self._evaluate(plan, now)

    def _evaluate(self, plan, now):
        identifier = plan["registration"]["id"]
        artifact = self._path(plan["outputs"]["artifact"])
        rows = self._path(plan["outputs"]["rows"])
        self._artifact(artifact, plan["outputs"], identifier)
        self._rows(rows, self._path(self.config["validation_truth"]))
        state_store.transact(self.root, "update", {}, identifier, artifact)
        item = state_store.candidate(state_store.load(self.root)[1], identifier)
        predictions = self._path(plan["outputs"]["predictions"]) if plan["outputs"].get("predictions") else artifact
        kind = plan["outputs"].get("kind", "file")
        self._validator(item, artifact, rows, now, predictions=predictions, kind=kind)
        baseline_units = self.root / "managed-baseline-units.csv"
        units = self.root / ("managed-" + identifier + "-units.csv")
        baseline = self._path(self.config["baseline_rows"])
        self._units(baseline, baseline_units)
        scored = self._units(rows, units)
        argv = ["gate", "--ledger", str(self.root / "ledger.json"), "--baseline", str(baseline_units),
                "--candidate", str(units), "--artifact", str(artifact), "--candidate-id", identifier,
                "--baseline-rows", str(baseline), "--candidate-rows", str(rows)]
        for key, value in self.config.get("gate", {}).items():
            argv.extend(["--" + key.replace("_", "-"), str(value)])
        if self.config.get("unit_metric"):
            argv += ["--unit-metric", self.config["unit_metric"]]
        if plan["outputs"].get("replication"):
            argv += ["--replication", str(self._path(plan["outputs"]["replication"]))]
        if self.config.get("scorer"):
            argv += ["--scorer", self._scorer()]
        args = analysis.parser().parse_args(argv)
        gate = args.run(args)
        sha = state_store.digest(artifact)
        state_store.transact(self.root, "score", {"artifact_sha256": sha, "oof": scored["overall"],
                             "selection_units": {u["unit"]: u["score"] for u in scored["units"]},
                             "gate": gate, "gate_artifact_sha256": sha}, identifier, artifact)
        self.state["active_plan"] = None
        self.state["history"].append({"candidate": identifier, "gate": gate["decision"], "at": now.isoformat()})
        if gate["decision"] != "PASS":
            self.state["no_gain"] += 1
            self._save()
            return
        state_store.transact(self.root, "update", {"status": "promoted"}, identifier)
        self.state["queue"].append({"id": identifier, "path": str(artifact), "sha256": sha,
                                    "private": True, "kind": kind, "predictions_path": str(predictions), "stage": "upload"})
        self._save()

    def _queue(self):
        for item in self.state["queue"]:
            if item["stage"] == "complete":
                continue
            if state_store.digest(item["path"]) != item["sha256"]:
                raise ManagedError("queued artifact bytes changed")
            config, ledger = state_store.load(self.root)
            current = state_store.candidate(ledger, item["id"])
            if state_store.evidence_errors(config, ledger, current):
                raise ManagedError("queued candidate evidence became stale")
            if item.get("operation_id"):
                result = self.adapter.reconcile(item["operation_id"])
            elif item["stage"] == "upload":
                result = self.adapter.upload(item)
            elif item["stage"] == "submit":
                result = self.adapter.submit(item)
            else:
                result = self.adapter.score(item["submission_ref"])
            if result.get("status") in ("unknown", "pending", "queued", "running"):
                item["operation_id"] = result.get("operation_id")
                self._save()
                continue
            if result.get("status") not in ("complete", "succeeded"):
                raise ManagedError("platform operation blocked or failed")
            item.pop("operation_id", None)
            if item["stage"] == "upload":
                item["upload_ref"] = result["upload_ref"]
                item["stage"] = "submit"
            elif item["stage"] == "submit":
                item["submission_ref"] = result["submission_ref"]
                item["stage"] = "score"
            else:
                if result.get("candidate_id") != item["id"] or result.get("artifact_sha256") != item["sha256"] or result.get("submission_ref") != item["submission_ref"]:
                    raise ManagedError("platform score receipt does not bind candidate, bytes and submission")
                score = analysis.finite(result["score"], "platform score")
                state_store.transact(self.root, "score", {"artifact_sha256": item["sha256"], "public": score}, item["id"], item["path"])
                item["stage"] = "complete"
                local = current["oof"]
                sign = 1 if config["metric"]["direction"] == "higher" else -1
                improved = self.state["best_score"] is None or sign * (local - self.state["best_score"]) > 0
                self.state["no_gain"] = 0 if improved else self.state["no_gain"] + 1
                if improved:
                    self.state["best_score"] = local
            self._save()

    def _finalize(self, now):
        self.state["frozen"] = True
        self.state["phase"] = "finalize"
        self._save()
        config, ledger = state_store.load(self.root)
        if any(item["stage"] != "complete" for item in self.state["queue"]):
            return
        if not ledger["holdout"].get("opened_at"):
            self.state["frozen_hashes"] = {item["id"]: item["artifact_sha256"] for item in ledger["candidates"] if item["status"] == "promoted"}
            if not self.state["frozen_hashes"]:
                raise ManagedError("no promoted candidates available for final selection")
            self._save()
            state_store.transact(self.root, "holdout-open", {"reason": "scheduled freeze; no further adaptive experiments"}, artifact=self._path(self.config["holdout_ids"]))
            self._save()
        if not self.state.get("holdout_complete"):
            plan = self._worker("holdout", now, selected_candidates=self.state["frozen_hashes"])
            if plan["kind"] != "holdout" or set(plan.get("outputs", {})) != set(self.state["frozen_hashes"]):
                raise ManagedError("holdout worker must score every frozen promoted candidate")
            self._execute(plan, "holdout")
            for identifier, sha in self.state["frozen_hashes"].items():
                item = state_store.candidate(state_store.load(self.root)[1], identifier)
                queued = next(q for q in self.state["queue"] if q["id"] == identifier)
                if item["artifact_sha256"] != sha or state_store.digest(queued["path"]) != sha:
                    raise ManagedError("candidate changed after freeze")
                path = self._path(plan["outputs"][identifier])
                self._rows(path, self._path(self.config["holdout_truth"]))
                self._validator(item, Path(queued["path"]), path, now, holdout=True,
                                predictions=queued.get("predictions_path"), kind=queued["kind"])
                value = self._score_rows(path)
                state_store.transact(self.root, "score", {"artifact_sha256": sha, "holdout": value["overall"]}, identifier, queued["path"])
            self.state["holdout_complete"] = True
            self._save()
        args = analysis.parser().parse_args(["trust", "--ledger", str(self.root / "ledger.json"), "--local", "holdout"])
        trust = args.run(args)
        state_store.transact(self.root, "trust", trust)
        gate = {"decision": "PASS", "reason": "frozen artifact identities and sealed holdout scores checked; selection count checked below"}
        state_store.transact(self.root, "config", {"gates": {"finals": gate}})
        # Config writes invalidate trust; compute it from the final configuration.
        trust = args.run(args)
        state_store.transact(self.root, "trust", trust)
        count = self.state["platform"]["rules"]["final_selection_count"]
        select = ["select", "--ledger", str(self.root / "ledger.json"), "--count", str(count),
                  "--residual-baseline", str(self.root / "managed-baseline-units.csv")]
        args = analysis.parser().parse_args(select)
        chosen = args.run(args)
        if chosen["status"] == "blocked" or chosen["unfilled"] or len(chosen["candidate_ids"]) != count:
            raise ManagedError("final selection has unfilled slots or blocked/diversity evidence")
        final = {**chosen, "reason": "scheduled analysis selection using sealed holdout and actual platform score receipts",
                 "holdout_decision": {"decision": "used", "reason": "frozen candidates scored on presealed independent units"}}
        state_store.transact(self.root, "final", final)
        refs = [next(item["submission_ref"] for item in self.state["queue"] if item["id"] == identifier) for identifier in chosen["candidate_ids"]]
        if self.state.get("final_operation_id"):
            result = self.adapter.reconcile(self.state["final_operation_id"])
        else:
            result = self.adapter.select_final(refs)
        if result.get("status") in ("unknown", "pending", "queued"):
            self.state["final_operation_id"] = result.get("operation_id")
            self._save()
            return
        if result.get("status") not in ("succeeded", "complete"):
            raise ManagedError("platform final selection not confirmed")
        readback = self.adapter.final_selection()
        if set(readback.get("refs", [])) != set(refs) or len(readback.get("refs", [])) != count:
            raise ManagedError("actual platform final selection differs from analysis")
        self.state.update(status="complete", phase="complete", final_readback=readback, next_due_at=None)
        self._save()

    def _tick(self, now):
        budget = self.store.inspect()
        if budget["status"] != "active" or budget["expired"]:
            self.state.update(status="blocked", reason="mandate inactive or expired")
            return
        platform = self.adapter.inspect()
        self.state["platform"] = platform
        if platform.get("status") == "blocked":
            self.state.update(status="blocked", reason=platform.get("reason", "platform capabilities unavailable"))
            return
        rules = platform["rules"]
        if rules.get("current") is not True or not isinstance(rules.get("final_selection_count"), int) or rules["final_selection_count"] < 1:
            raise ManagedError("current rules and actual final selection count required")
        acceptance = budget["mandate"]["rules_acceptance"]
        if not acceptance["allowed"] or acceptance.get("mode", "pinned") == "pinned" and acceptance["document_sha256"] != rules.get("sha256", rules.get("hash")):
            self.state.update(status="blocked", reason="current rules version was not approved in initial mandate")
            return
        if not platform.get("joined") or platform.get("joined_rules_hash") != rules.get("sha256", rules.get("hash")):
            if self.state.get("join_operation_id"):
                joined = self.adapter.reconcile(self.state["join_operation_id"])
            else:
                joined = self.adapter.join(rules.get("sha256", rules.get("hash")))
            if joined.get("status") in ("unknown", "pending", "queued"):
                self.state["join_operation_id"] = joined.get("operation_id")
                self.state["reason"] = "rules acceptance awaiting authoritative reconciliation"
                return
            if joined.get("status") not in ("complete", "succeeded"):
                self.state.update(status="blocked", reason="rules join not confirmed")
                return
            self.state.pop("join_operation_id", None)
        if self.state["intel_due_at"] is None or now >= instant(self.state["intel_due_at"]):
            delta = self.adapter.refresh_intel(self.state["intel_cursor"])
            if delta.get("status") == "blocked":
                raise ManagedError("community refresh capability unavailable")
            changed = bool(delta.get("changed", delta.get("delta", delta.get("events"))))
            self.state["intel"] = delta
            self.state["intel_cursor"] = delta.get("cursor", self.state["intel_cursor"])
            self.state["intel_due_at"] = (now + timedelta(seconds=self.config.get("intel_interval_seconds", 3600))).isoformat()
            if changed and self.state["phase"] != "bootstrap":
                self.state["reevaluate"] = True
            self._save()
        if self.state["phase"] == "bootstrap":
            self._bootstrap(now)
            return
        if self.state.get("active_plan"):
            jobs = [job for job in self.state["jobs"] if job["identity"] == {"task": "experiment", "attempt": self.state["attempt"]}]
            receipts = {receipt["id"]: receipt for receipt in self.store.inspect()["receipts"]}
            if not jobs or receipts[jobs[-1]["id"]]["status"] != "complete":
                if jobs and jobs[-1].get("process_path") and self._path(jobs[-1]["process_path"]).exists():
                    process = state_store.read_json(self._path(jobs[-1]["process_path"]))
                    receipt = receipts[jobs[-1]["id"]]
                    expires = instant(process["started_at"]) + timedelta(seconds=receipt["reserved_costs"]["wall_seconds"])
                    if receipt["status"] == "pending" and instant() < expires:
                        try:
                            os.kill(process["pid"], 0)
                        except ProcessLookupError:
                            pass
                        else:
                            self.state["reason"] = "reserved training supervisor is still running"
                            return
                raise ManagedError("interrupted training requires local job reconciliation; no automatic retraining")
            self._evaluate(self.state["active_plan"], now)
            return
        remaining = self._remaining(now)
        if remaining is not None and remaining <= 0:
            self.state.update(status="blocked", reason="deadline passed before verified final selection")
            return
        self._queue()
        if any(item["stage"] != "complete" for item in self.state["queue"]):
            return
        remaining_budget = self.store.inspect()["remaining"]
        exhausted = remaining_budget.get("train_runs", 1) <= 0
        if self._route(now) == "freeze_and_finalize" or exhausted or self.state["no_gain"] >= self.config.get("max_no_gain", 3) or self.state.get("stop_requested"):
            self._finalize(now)
            return
        self.state["attempt"] += 1
        self._save()
        plan = self._worker("reevaluate" if self.state.pop("reevaluate", False) else "plan", now)
        if plan["kind"] == "stop":
            self.state["stop_requested"] = plan["reason"]
        elif plan["kind"] == "wait":
            self.state["next_due_at"] = (now + timedelta(seconds=max(1, min(float(plan.get("seconds", 60)), 86400)))).isoformat()
        elif plan["kind"] == "experiment":
            self._candidate(plan, now)
        else:
            raise ManagedError("iteration worker must return experiment, wait, or stop")

    def tick(self, now=None):
        now = instant(now)
        with self._lock():
            self.state = self._read()
            self.config.update(self.state.get("setup", {}))
            if self.user_goal is not None:
                self.config["goal"] = self.user_goal
            if self.state["status"] == "complete":
                return self.state
            if self.state["next_due_at"] and now < instant(self.state["next_due_at"]):
                return self.state
            self.state["status"] = "running"
            self.state["next_due_at"] = (now + timedelta(seconds=self.config.get("interval_seconds", 60))).isoformat()
            self._save()
            try:
                self._recover_jobs()
                self._tick(now)
                if self.state["status"] == "running":
                    self.state["status"] = "ready"
                    self.state["errors"] = 0
            except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
                self.state["errors"] += 1
                self.state["reason"] = str(error)
                self.state["status"] = "blocked" if self.state["errors"] >= self.config.get("max_errors", 3) else "retry_wait"
                self.state["next_due_at"] = (now + timedelta(seconds=min(3600, self.config.get("interval_seconds", 60) * 2 ** self.state["errors"]))).isoformat()
            self._save()
            return self.state

    def inspect(self):
        return {"state": self._read(), "budget": self.store.inspect(),
                "worker_command": self.config["worker_command"], "installed": (self.root / "managed-service.json").exists()}

    def serve(self, max_ticks=None, max_seconds=None):
        start = time.monotonic()
        ticks = 0
        while max_ticks is None or ticks < max_ticks:
            if max_seconds is not None and time.monotonic() - start >= max_seconds:
                break
            state = self.tick()
            ticks += 1
            if state["status"] in ("complete", "blocked"):
                break
            delay = max(0, (instant(state["next_due_at"]) - instant()).total_seconds())
            if max_seconds is not None:
                delay = min(delay, max(0, max_seconds - (time.monotonic() - start)))
            # Short sleeps allow shutdown signals and bounded service invocations.
            while delay > 0:
                chunk = min(delay, 30)
                time.sleep(chunk)
                delay -= chunk
        return self._read()


def install(config_path, backend="pid"):
    config = state_store.read_json(config_path)
    root = Path(config["root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".managed-install.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        return _install(config_path, backend)


def _install(config_path, backend="pid"):
    """Start a self-contained process or install an explicit user-level scheduler."""
    config_path = Path(config_path).resolve()
    config = state_store.read_json(config_path)
    root = Path(config["root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "managed-service.json"
    argv = [sys.executable, str(Path(__file__).resolve()), "serve", "--config", str(config_path)]
    if manifest.exists():
        previous = state_store.read_json(manifest)
        if previous.get("backend") == "pid":
            try:
                os.kill(previous["pid"], 0)
            except ProcessLookupError:
                manifest.unlink()
            else:
                raise ManagedError("managed service already running")
        else:
            raise ManagedError("scheduler already installed; uninstall first")
    label = "org.codex.kaggle." + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    if backend == "pid":
        with (root / "managed-service.log").open("a") as log:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        result = {"backend": backend, "pid": process.pid, "argv": argv}
    elif backend == "launchd":
        import plistlib
        path = Path.home() / "Library" / "LaunchAgents" / (label + ".plist")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ManagedError("scheduler file already exists")
        path.write_bytes(plistlib.dumps({"Label": label, "ProgramArguments": argv, "RunAtLoad": True,
                                        "KeepAlive": {"SuccessfulExit": False}, "WorkingDirectory": str(root)}))
        subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)], check=True)
        result = {"backend": backend, "path": str(path), "label": label}
    elif backend == "systemd":
        path = Path.home() / ".config" / "systemd" / "user" / (label + ".service")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ManagedError("scheduler file already exists")
        if any("\n" in v or '"' in v or "%" in v for v in argv):
            raise ManagedError("unsupported scheduler argv characters")
        path.write_text("[Unit]\nDescription=Managed Kaggle worker\n[Service]\nType=simple\nExecStart=" + " ".join('"' + v.replace("\\", "\\\\") + '"' for v in argv) + "\nRestart=on-failure\n[Install]\nWantedBy=default.target\n")
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", "--now", path.name], check=True)
        result = {"backend": backend, "path": str(path), "label": label}
    else:
        raise ManagedError("unknown scheduler backend")
    state_store.atomic_json(manifest, result)
    return result


def uninstall(root):
    manifest = Path(root) / "managed-service.json"
    state = state_store.read_json(manifest)
    if state["backend"] == "pid":
        try:
            os.kill(state["pid"], 0)
        except ProcessLookupError:
            manifest.unlink()
            return {"uninstalled": True, "already_stopped": True}
        try:
            # Verify argv before signalling a potentially recycled PID.
            actual = subprocess.check_output(["ps", "-p", str(state["pid"]), "-o", "command="], text=True)
            if "managed.py serve" not in actual or state["argv"][-1] not in actual:
                raise ManagedError("PID no longer belongs to this configured service")
            os.killpg(state["pid"], signal.SIGTERM)
        except subprocess.CalledProcessError:
            pass
    elif state["backend"] == "launchd":
        subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}", state["path"]], check=True)
        Path(state["path"]).unlink()
    else:
        subprocess.run(["systemctl", "--user", "disable", "--now", state["label"] + ".service"], check=True)
        Path(state["path"]).unlink()
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    manifest.unlink()
    return {"uninstalled": True}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_job":
        try:
            print(json.dumps(supervised_job(Path(argv[1])), allow_nan=False))
            return 0
        except (ValueError, OSError, subprocess.SubprocessError):
            return 2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["tick", "serve", "inspect", "install", "uninstall"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-ticks", type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--backend", choices=["pid", "launchd", "systemd"], default="pid")
    args = parser.parse_args(argv)
    try:
        config = state_store.read_json(args.config)
        if args.action == "install":
            result = install(args.config, args.backend)
        elif args.action == "uninstall":
            result = uninstall(config["root"])
        else:
            runner = ManagedRunner(config)
            if args.action == "serve":
                result = runner.serve(args.max_ticks, args.max_seconds)
            else:
                result = getattr(runner, args.action)()
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
