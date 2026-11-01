"""Atomic, local-only competition facts and decision prerequisites."""

import hashlib
import json
import math
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import fcntl


GATES = ("strategy", "test_distribution", "validation", "midpoint", "finals")
SCORES = ("oof", "holdout", "public", "private")
SHA = re.compile(r"[0-9a-f]{64}\Z")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    def reject(value):
        raise ValueError(f"non-finite JSON number: {value}")
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject, object_pairs_hook=unique)


def atomic_json(path, value):
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def locked(root):
    with (root / ".workspace.lock").open("a", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def legacy(root):
    return [name for name in ("GAMEPLAN.md", "competition_ledger.json") if (root / name).exists()]


def require_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} requires a non-empty string")
    return value


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def require_sha(value):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError("artifact SHA256 must be 64 lowercase hexadecimal characters")
    return value


def metric_valid(metric):
    return (isinstance(metric, dict) and isinstance(metric.get("name"), str)
            and bool(metric["name"].strip()) and metric.get("direction") in ("higher", "lower"))


def blank_ledger(metric):
    return {
        "schema": 1, "metric": metric, "public_units": None,
        "validator": {"authority": None, "covered_stages": []},
        "findings": [], "candidates": [], "trust": None,
        "holdout": {"opened_at": None, "sha256": None, "audit": []},
        "final_selection": None, "audit": [],
    }


def load(root):
    if (root / ".state-transaction.json").exists():
        raise ValueError("recovery_required: interrupted config transaction; run recover")
    if not (root / "ledger.json").exists() and legacy(root):
        raise ValueError("legacy_not_migrated: preserve old work and explicitly migrate before writing")
    config = read_json(root / "competition.json")
    ledger = read_json(root / "ledger.json")
    if config.get("schema") != 1 or ledger.get("schema") != 1:
        raise ValueError("unsupported workspace schema")
    if not ledger.get("candidates") or not metric_valid(ledger.get("metric")):
        ledger["metric"] = config.get("metric")
    if ledger.get("metric") != config.get("metric"):
        raise ValueError("competition and ledger metric disagree; explicitly reconcile before writing")
    if (config.get("public_units") is not None and config.get("budget", {}).get("public_units") is not None
            and config["public_units"] != config["budget"]["public_units"]):
        raise ValueError("public_units disagrees with legacy budget.public_units")
    ledger["public_units"] = config.get("public_units") if config.get("public_units") is not None else config.get("budget", {}).get("public_units")
    return config, ledger


def recover(root):
    journal = root / ".state-transaction.json"
    if not journal.exists():
        return False
    transaction = read_json(journal)
    if set(transaction) != {"competition", "ledger"} or any(not isinstance(transaction[key], dict) or transaction[key].get("schema") != 1 for key in transaction):
        raise ValueError("invalid config recovery journal; preserve and inspect it manually")
    atomic_json(root / "competition.json", transaction["competition"])
    atomic_json(root / "ledger.json", transaction["ledger"])
    journal.unlink()
    return True


def candidate(ledger, identifier):
    for item in ledger["candidates"]:
        if item["id"] == identifier:
            return item
    raise ValueError(f"candidate is not preregistered: {identifier}")


def blockers(ledger):
    return [item["id"] for item in ledger["findings"]
            if item.get("severity") in ("blocker", "critical", "high") and item.get("status") == "open"]


def unit_scores(value):
    if not isinstance(value, dict) or any(not isinstance(key, str) or not key.strip()
                                        or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
                                        for key, score in value.items()):
        raise ValueError("selection_units must map non-empty unit IDs to finite scores")
    return value


def gate_pass(gate):
    return isinstance(gate, dict) and gate.get("decision") == "PASS"


def evidence_errors(config, ledger, item):
    errors = []
    if not metric_valid(config.get("metric")):
        errors.append("metric is not configured")
    if not item.get("artifact_sha256"):
        errors.append("candidate artifact is not registered")
    if item.get("gate_artifact_sha256") != item.get("artifact_sha256") or not gate_pass(item.get("gate")):
        errors.append("candidate gate is missing, blocked, or stale")
    gate = item.get("gate") or {}
    evidence = gate.get("evidence") or {}
    if evidence.get("metric") != config.get("metric"):
        errors.append("gate metric is stale or missing")
    if gate.get("artifact_sha256") != item.get("artifact_sha256"):
        errors.append("gate artifact hash does not match")
    if gate.get("validator_authority") != "full":
        errors.append("gate validator authority is not full")
    if not set(item["stages"]).issubset(set(gate.get("covered_stages", []))):
        errors.append("gate does not cover candidate stages")
    if item["swept"] > 1:
        trials = evidence.get("search_trials")
        replication = evidence.get("replication") or {}
        if isinstance(trials, bool) or not isinstance(trials, int) or trials < item["swept"]:
            errors.append("gate search_trials understates preregistered swept trials")
        if (replication.get("fresh") is not True or replication.get("independent") is not True
                or replication.get("candidate_sha256") != item["artifact_sha256"]
                or not isinstance(replication.get("evidence_id"), str) or not replication["evidence_id"].strip()):
            errors.append("swept candidate requires fresh independent replication bound to current artifact")
    if item.get("oof") is None:
        errors.append("OOF evidence is missing")
    validator = ledger["validator"]
    if validator.get("authority") != "full":
        errors.append("validator authority is not full")
    if not set(item["stages"]).issubset(set(validator.get("covered_stages", []))):
        errors.append("validator does not cover candidate stages")
    if not item["stages"]:
        errors.append("candidate stages are missing")
    if blockers(ledger):
        errors.append("open blocker findings")
    for name in ("strategy", "test_distribution", "validation"):
        if not gate_pass(config.get("gates", {}).get(name)):
            errors.append(f"{name} gate has not passed")
    return errors


def invalidate(ledger, item, reason):
    item["gate"] = None
    item["gate_artifact_sha256"] = None
    if item["status"] == "promoted":
        item["status"] = "registered"
    ledger["trust"] = None
    ledger["final_selection"] = None
    ledger["audit"].append({"event": "evidence_invalidated", "id": item["id"], "reason": reason, "at": stamp()})


def transact(root, action, payload=None, identifier=None, artifact=None):
    payload = {} if payload is None else payload
    if not isinstance(payload, dict):
        raise ValueError("input must be a JSON object")
    with locked(root):
        recovered = recover(root)
        if action == "recover":
            return {"recovered": recovered}
        config, ledger = load(root)
        if action == "config":
            return configure(root, config, ledger, payload)
        result = apply(config, ledger, action, payload, identifier, artifact)
        ledger["audit"].append({"event": action, "id": identifier, "at": stamp()})
        atomic_json(root / "ledger.json", ledger)
        return result


def configure(root, config, ledger, payload):
    allowed = {"competition", "metric", "gates", "deadline", "reserve_hours", "public_units", "budget",
               "route", "goal", "validation", "next_action", "late_join_days", "started_at"}
    if not payload or set(payload) - allowed:
        raise ValueError("config contains unknown fields or is empty")
    if "metric" in payload:
        if not metric_valid(payload["metric"]) or set(payload["metric"]) != {"name", "direction"}:
            raise ValueError("metric requires name and direction higher or lower")
        if ledger["candidates"] and payload["metric"] != config["metric"]:
            raise ValueError("cannot change metric after experiment registration")
    for field in ("deadline", "started_at"):
        if payload.get(field) is not None:
            value = datetime.fromisoformat(payload[field].replace("Z", "+00:00"))
            if value.tzinfo is None:
                raise ValueError(f"{field} requires explicit timezone")
    for field in ("reserve_hours", "late_join_days", "public_units"):
        value = payload.get(field)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            raise ValueError(f"{field} must be finite and non-negative")
    if "gates" in payload:
        gates = payload["gates"]
        if not isinstance(gates, dict) or set(gates) - set(GATES):
            raise ValueError("unknown competition gate")
        for name, value in gates.items():
            if value is not None:
                if not isinstance(value, dict) or value.get("decision") not in ("PASS", "BLOCK"):
                    raise ValueError(f"{name} requires PASS or BLOCK")
                require_text(value.get("reason"), f"{name} gate reason")
        payload["gates"] = dict(config["gates"], **gates)
    if "budget" in payload:
        budget = payload["budget"]
        if not isinstance(budget, dict) or set(budget) - {"gpu_hours", "money", "submissions", "public_units"}:
            raise ValueError("unknown budget field")
        for value in budget.values():
            if value is not None and (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0):
                raise ValueError("budget amounts must be finite and non-negative")
        payload["budget"] = dict(config["budget"], **budget)
    updated = dict(config, **payload)
    if (updated.get("public_units") is not None and updated["budget"].get("public_units") is not None
            and updated["public_units"] != updated["budget"]["public_units"]):
        raise ValueError("public_units disagrees with legacy budget.public_units")
    updated["updated_at"] = stamp()
    ledger["metric"] = updated["metric"]
    ledger["public_units"] = updated.get("public_units") if updated.get("public_units") is not None else updated["budget"].get("public_units")
    ledger["trust"] = None
    ledger["final_selection"] = None
    for item in ledger["candidates"]:
        if item["status"] == "promoted" and evidence_errors(updated, ledger, item):
            item["status"] = "registered"
    ledger["audit"].append({"event": "config", "fields": sorted(payload), "at": stamp()})
    atomic_json(root / ".state-transaction.json", {"competition": updated, "ledger": ledger})
    recover(root)
    return {"configured": sorted(payload)}


def apply(config, ledger, action, payload, identifier, artifact):
    if action == "register":
        allowed = {"id", "family", "parent", "root", "structural", "hypothesis", "swept", "stages", "selection_units"}
        if set(payload) - allowed:
            raise ValueError("registration contains results or unknown fields; preregister before results")
        identifier = require_text(payload.get("id"), "id")
        if any(item["id"] == identifier for item in ledger["candidates"]):
            raise ValueError("candidate already registered")
        for field in ("family", "hypothesis"):
            require_text(payload.get(field), field)
        if not isinstance(payload.get("structural"), bool):
            raise ValueError("structural must be a boolean")
        swept = 1 if isinstance(payload.get("swept"), bool) else payload.get("swept")
        if not isinstance(swept, int) or swept < 1:
            raise ValueError("swept must be an integer >= 1 (legacy booleans count as one)")
        stages = payload.get("stages", [])
        if not isinstance(stages, list) or not stages or any(not isinstance(v, str) or not v.strip() for v in stages):
            raise ValueError("stages must list the complete candidate pipeline")
        parent = payload.get("parent")
        root = identifier
        if parent is not None:
            root = candidate(ledger, parent)["root"]
        if payload.get("root", root) != root:
            raise ValueError("root must match parent lineage")
        units = unit_scores(payload.get("selection_units", {}))
        if units:
            raise ValueError("preregister before selection-unit results")
        item = {"id": identifier, "family": payload["family"], "parent": parent, "root": root,
                "structural": payload["structural"], "hypothesis": payload["hypothesis"],
                "swept": swept, "stages": stages,
                "artifact_sha256": None, "gate_artifact_sha256": None, "gate": None,
                "oof": None, "holdout": None, "public": None, "private": None,
                "status": "registered", "selection_units": units, "registered_at": stamp()}
        ledger["candidates"].append(item)
        ledger["trust"] = None
        ledger["final_selection"] = None
        return {"registered": identifier}
    if action == "update":
        item = candidate(ledger, identifier)
        if set(payload) - {"stages", "status", "selection_units"}:
            raise ValueError("update permits only stages, status, selection_units, and --artifact")
        if "stages" in payload:
            stages = payload["stages"]
            if not isinstance(stages, list) or not stages or any(not isinstance(v, str) or not v.strip() for v in stages):
                raise ValueError("stages must be a non-empty string list")
            if stages != item["stages"]:
                invalidate(ledger, item, "pipeline stages changed")
                item["stages"] = stages
        if artifact is not None:
            current = digest(artifact)
            if current != item["artifact_sha256"]:
                invalidate(ledger, item, "artifact bytes changed")
                for name in SCORES:
                    item[name] = None
                item["selection_units"] = {}
                item["artifact_sha256"] = current
        if "selection_units" in payload:
            item["selection_units"] = unit_scores(payload["selection_units"])
            ledger["trust"] = None
            ledger["final_selection"] = None
        if "status" in payload:
            if payload["status"] not in ("registered", "running", "failed", "withdrawn", "promoted"):
                raise ValueError("invalid candidate status")
            if payload["status"] == "promoted":
                errors = evidence_errors(config, ledger, item)
                if errors:
                    raise ValueError("cannot promote: " + "; ".join(errors))
            item["status"] = payload["status"]
            ledger["final_selection"] = None
        return {"updated": identifier, "artifact_sha256": item["artifact_sha256"]}
    if action == "score":
        item = candidate(ledger, identifier)
        if set(payload) - set(SCORES) - {"gate", "gate_artifact_sha256", "artifact_sha256", "selection_units"}:
            raise ValueError("score contains unknown fields")
        current = require_sha(payload.get("artifact_sha256"))
        if current != item["artifact_sha256"]:
            raise ValueError("score artifact does not match registered current artifact")
        if artifact is not None and digest(artifact) != current:
            raise ValueError("artifact bytes changed since score was produced")
        for name in SCORES:
            if name in payload:
                value = payload[name]
                number = value
                if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number):
                    raise ValueError(f"{name} requires a finite score")
                if name == "holdout" and not ledger["holdout"].get("opened_at"):
                    raise ValueError("sealed holdout must be explicitly opened before scoring")
                item[name] = value
        if "selection_units" in payload:
            item["selection_units"] = unit_scores(payload["selection_units"])
        if "gate" in payload:
            gate = payload["gate"]
            if not isinstance(gate, dict) or gate.get("decision") not in ("PASS", "BLOCK"):
                raise ValueError("gate requires decision PASS or BLOCK")
            if not gate.get("evidence") or not isinstance(gate["evidence"], dict):
                raise ValueError("gate requires non-empty evidence from the statistical analysis")
            gate_sha = payload.get("gate_artifact_sha256", gate.get("artifact_sha256"))
            if gate_sha != current or gate.get("artifact_sha256") != current:
                raise ValueError("gate evidence must be bound to the current artifact")
            item["gate"] = gate
            item["gate_artifact_sha256"] = current
        if item["status"] == "promoted" and evidence_errors(config, ledger, item):
            item["status"] = "registered"
        ledger["trust"] = None
        ledger["final_selection"] = None
        return {"scored": identifier}
    if action == "finding":
        identifier = require_text(payload.get("id"), "finding id")
        if any(item["id"] == identifier for item in ledger["findings"]):
            raise ValueError("finding already exists")
        if payload.get("severity") not in ("blocker", "critical", "high", "warning", "info"):
            raise ValueError("severity must be blocker, critical, high, warning, or info")
        require_text(payload.get("description"), "description")
        ledger["findings"].append({"id": identifier, "severity": payload["severity"],
                                   "description": payload["description"], "status": "open", "created_at": stamp()})
        ledger["trust"] = None
        ledger["final_selection"] = None
        if payload["severity"] in ("blocker", "critical", "high"):
            for item in ledger["candidates"]:
                if item["status"] == "promoted":
                    item["status"] = "registered"
        return {"finding": identifier}
    if action == "resolve":
        finding = next((item for item in ledger["findings"] if item["id"] == identifier), None)
        if finding is None or finding["status"] != "open":
            raise ValueError("finding is missing or already resolved")
        if payload.get("status") not in ("resolved", "accepted"):
            raise ValueError("resolution status must be resolved or accepted")
        reason = require_text(payload.get("reason"), "resolution reason")
        finding.update(status=payload["status"], reason=reason, resolved_at=stamp())
        ledger["trust"] = None
        ledger["final_selection"] = None
        return {"resolved": identifier}
    if action == "validator":
        if payload.get("authority") not in ("full", "partial", "proxy", "unknown", "veto_only"):
            raise ValueError("validator authority must be full, partial, proxy, unknown, or veto_only")
        stages = payload.get("covered_stages")
        if not isinstance(stages, list) or not stages or any(not isinstance(v, str) or not v.strip() for v in stages):
            raise ValueError("validator covered_stages must be a non-empty string list")
        ledger["validator"] = payload
        ledger["trust"] = None
        ledger["final_selection"] = None
        for item in ledger["candidates"]:
            if item["status"] == "promoted" and evidence_errors(config, ledger, item):
                item["status"] = "registered"
        return {"validator_saved": True}
    if action == "trust":
        if blockers(ledger):
            raise ValueError("cannot save trust with open blockers")
        if payload.get("regime") not in ("ALIGNED", "WEAK", "DIVERGENT", "INSUFFICIENT"):
            raise ValueError("trust regime must be ALIGNED, WEAK, DIVERGENT, or INSUFFICIENT")
        ids = payload.get("candidate_ids")
        if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
            raise ValueError("trust requires unique candidate_ids")
        for value in ids:
            candidate(ledger, value)
        require_text(payload.get("reason"), "trust reason")
        ledger["trust"] = dict(payload, saved_at=stamp(), facts_sha256=trust_fingerprint(ledger, config))
        ledger["final_selection"] = None
        return {"trust_saved": True}
    if action == "holdout-register":
        if artifact is None:
            raise ValueError("holdout-register requires --artifact containing sealed unit IDs")
        if ledger["holdout"].get("sha256") is not None:
            raise ValueError("holdout IDs are already sealed; cannot replace")
        if ledger["holdout"].get("opened_at"):
            raise ValueError("holdout already opened")
        ledger["holdout"]["sha256"] = digest(artifact)
        ledger["holdout"]["audit"].append({"event": "sealed", "at": stamp()})
        ledger["final_selection"] = None
        return {"holdout_registered": True, "sha256": ledger["holdout"]["sha256"]}
    if action == "holdout-open":
        holdout = ledger["holdout"]
        if holdout.get("opened_at"):
            raise ValueError("holdout already opened; second opening is forbidden")
        if artifact is None or not holdout.get("sha256") or digest(artifact) != holdout["sha256"]:
            raise ValueError("holdout IDs do not match sealed SHA256")
        reason = require_text(payload.get("reason"), "holdout opening reason")
        if blockers(ledger):
            raise ValueError("cannot open holdout with open blockers")
        holdout["opened_at"] = stamp()
        holdout["audit"].append({"event": "opened", "at": holdout["opened_at"], "reason": reason})
        ledger["trust"] = None
        ledger["final_selection"] = None
        return {"holdout_opened_at": holdout["opened_at"]}
    if action == "final":
        if payload.get("status") == "blocked":
            raise ValueError("cannot save a blocked analysis selection")
        if blockers(ledger):
            raise ValueError("cannot save final selection with open blockers")
        ids = payload.get("candidate_ids")
        if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
            raise ValueError("final requires unique candidate_ids")
        for value in ids:
            item = candidate(ledger, value)
            errors = evidence_errors(config, ledger, item)
            if item["status"] != "promoted" or errors:
                raise ValueError(f"final candidate {value} is not promoted with valid evidence: {'; '.join(errors)}")
            if item["public"] is None:
                raise ValueError(f"final candidate {value} has no returned public score receipt")
        if not gate_pass(config.get("gates", {}).get("finals")):
            raise ValueError("finals gate has not passed")
        decision = payload.get("holdout_decision")
        if not isinstance(decision, dict) or decision.get("decision") not in ("used", "not_opened", "not_used"):
            raise ValueError("final requires an explicit holdout_decision: used, not_opened, or not_used")
        require_text(decision.get("reason"), "holdout decision reason")
        if decision["decision"] == "used":
            if not ledger["holdout"].get("opened_at") or any(candidate(ledger, value)["holdout"] is None for value in ids):
                raise ValueError("used holdout requires opening audit and selected-candidate scores")
        elif decision["decision"] == "not_opened" and ledger["holdout"].get("opened_at"):
            raise ValueError("cannot claim not_opened after holdout opening")
        if not ledger["trust"] or ledger["trust"].get("facts_sha256") != trust_fingerprint(ledger, config):
            raise ValueError("current trust decision is missing or stale")
        if not set(ids).issubset(set(ledger["trust"]["candidate_ids"])):
            raise ValueError("final candidates were not covered by current trust evidence")
        require_text(payload.get("reason"), "final selection reason")
        ledger["final_selection"] = dict(payload, saved_at=stamp(), facts_sha256=trust_fingerprint(ledger, config))
        if ledger["trust"]["regime"] == "INSUFFICIENT":
            ledger["final_selection"]["evidence_basis"] = "local_only"
        return {"final_saved": True}
    raise ValueError(f"unsupported action: {action}")


def trust_fingerprint(ledger, config):
    facts = {key: ledger[key] for key in ("metric", "validator", "findings", "candidates", "holdout")}
    facts["config"] = {key: config.get(key) for key in ("metric", "gates", "public_units", "budget", "deadline", "reserve_hours")}
    return hashlib.sha256(json.dumps(facts, sort_keys=True, allow_nan=False).encode()).hexdigest()


def status(root, now):
    if (root / ".state-transaction.json").exists():
        return {"state": "recovery_required", "recommendations": ["recover_state"]}
    if (not (root / "competition.json").exists() or not (root / "ledger.json").exists()) and legacy(root):
        return {"state": "legacy_not_migrated", "legacy_files": legacy(root), "recommendations": ["explicit_migration_required"]}
    config, ledger = load(root)
    recommendations = []
    if blockers(ledger):
        recommendations.append("resolve_blockers")
    for name in GATES:
        if not gate_pass(config.get("gates", {}).get(name)):
            recommendations.append(f"complete_{name}_gate")
    hours = None
    if config.get("deadline"):
        deadline = datetime.fromisoformat(config["deadline"].replace("Z", "+00:00"))
        if deadline.tzinfo is None or now.tzinfo is None:
            raise ValueError("deadline and current time require explicit timezones")
        hours = (deadline - now).total_seconds() / 3600
        reserve = config.get("reserve_hours")
        if reserve is not None:
            if isinstance(reserve, bool) or not isinstance(reserve, (float, int)) or not math.isfinite(reserve) or reserve < 0:
                raise ValueError("reserve_hours must be finite and non-negative")
            if hours < reserve:
                recommendations.insert(0, "freeze_and_finalize")
        if hours <= 0:
            recommendations.insert(0, "deadline_passed")
        late_days = config.get("late_join_days", 7)
        if isinstance(late_days, bool) or not isinstance(late_days, (int, float)) or late_days < 0 or not math.isfinite(late_days):
            raise ValueError("late_join_days must be finite and non-negative")
        if hours <= late_days * 24 and not ledger["candidates"]:
            recommendations.append("late_join_build_minimal_valid_baseline")
        if config.get("started_at"):
            started = datetime.fromisoformat(config["started_at"].replace("Z", "+00:00"))
            if started.tzinfo is None:
                raise ValueError("started_at requires an explicit timezone")
            if now >= started + (deadline - started) / 2 and not gate_pass(config.get("gates", {}).get("midpoint")):
                recommendations.append("midpoint_review_due")
    structural_pairs = sum(bool(item["structural"] and item["oof"] is not None and item["public"] is not None)
                           for item in ledger["candidates"])
    trust = ledger["trust"]
    trust_current = bool(trust and trust.get("facts_sha256") == trust_fingerprint(ledger, config))
    if structural_pairs >= 5 and not trust_current:
        recommendations.append("refresh_cv_public_trust")
    recommendations.append("build_baseline" if not ledger["candidates"] else "iterate_registered_experiments")
    return {"state": "ready", "config": config, "ledger": ledger, "open_blockers": blockers(ledger),
            "hours_remaining": hours, "structural_pairs": structural_pairs, "trust_current": trust_current,
            "final_current": bool(ledger["final_selection"] and ledger["final_selection"].get("facts_sha256") == trust_fingerprint(ledger, config)),
            "candidate_errors": {item["id"]: evidence_errors(config, ledger, item) for item in ledger["candidates"]},
            "recommendations": recommendations}
