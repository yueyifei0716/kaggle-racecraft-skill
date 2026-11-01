"""Local competition state and append-only experiment events."""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_store


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def initialize(root):
    root.mkdir(parents=True, exist_ok=True)
    with state_store.locked(root):
        if state_store.legacy(root):
            raise ValueError("legacy_not_migrated: refusing to initialize over legacy work")
        if any((root / name).exists() for name in ("competition.json", "ledger.json", "experiments.jsonl", ".state-transaction.json")):
            raise ValueError("state or event log already exists; refusing to replace it")
        config = {
            "schema": 1, "competition": None, "deadline": None,
            "reserve_hours": None, "goal": None, "route": None,
            "budget": {"gpu_hours": None, "money": None, "submissions": None},
            "public_units": None, "metric": {"name": None, "direction": None},
            "gates": {name: None for name in state_store.GATES},
            "late_join_days": 7, "started_at": None,
            "validation": None, "holdout_opened_at": None, "next_action": None,
            "created_at": stamp(),
        }
        state_store.atomic_json(root / "competition.json", config)
        state_store.atomic_json(root / "ledger.json", state_store.blank_ledger(config["metric"]))
        (root / "experiments.jsonl").touch(exist_ok=False)


def record(root, path):
    with state_store.locked(root):
        state_store.recover(root)
        _record(root, path)


def _record(root, path):
    if not (root / "ledger.json").exists() and state_store.legacy(root):
        raise ValueError("legacy_not_migrated: cannot append before explicit migration")
    config = read_json(root / "competition.json")
    if config.get("schema") != 1:
        raise ValueError("unsupported workspace schema")
    event = state_store.read_json(path)
    if not isinstance(event, dict) or not all(
        isinstance(event.get(key), str) and event[key].strip()
        for key in ("id", "event")
    ):
        raise ValueError("event requires non-empty string id and event")
    event.setdefault("recorded_at", stamp())
    payload = json.dumps(event, ensure_ascii=False, allow_nan=False)
    log = root / "experiments.jsonl"
    previous = log.read_text(encoding="utf-8")
    for number, line in enumerate(previous.splitlines(), 1):
        if line.strip():
            parsed = json.loads(line)
            if not isinstance(parsed, dict) or not all(parsed.get(key) for key in ("id", "event")):
                raise ValueError(f"invalid existing event at line {number}")
    separator = "\n" if previous and not previous.endswith("\n") else ""
    with log.open("a", encoding="utf-8") as stream:
        stream.write(separator + payload + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def inspect(root, now):
    if (root / ".state-transaction.json").exists():
        return state_store.status(root, now)
    if (not (root / "competition.json").exists() or not (root / "ledger.json").exists()) and state_store.legacy(root):
        return state_store.status(root, now)
    config = read_json(root / "competition.json")
    events = []
    for number, line in enumerate((root / "experiments.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid event at line {number}") from exc
    latest = {}
    for event in events:
        latest[event["id"]] = event
    result = {"config": config, "event_count": len(events), "latest_events": latest}
    if config.get("deadline"):
        deadline = datetime.fromisoformat(config["deadline"].replace("Z", "+00:00"))
        if deadline.tzinfo is None or now.tzinfo is None:
            raise ValueError("deadline and current time require explicit timezones")
        hours = (deadline - now).total_seconds() / 3600
        result["hours_remaining"] = hours
        reserve = config.get("reserve_hours")
        if reserve is not None:
            if isinstance(reserve, bool) or not isinstance(reserve, (int, float)) or not math.isfinite(reserve) or reserve < 0:
                raise ValueError("reserve_hours must be a finite non-negative number")
            result["hours_before_safety_window"] = hours - reserve
        result["ended"] = hours <= 0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["init", "inspect", "record", "status", "config", "recover", "register", "update", "score",
                                           "finding", "resolve", "validator", "trust", "final",
                                           "holdout-register", "holdout-open"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--event", type=Path, help="JSON event file for record")
    parser.add_argument("--now", help="timezone-aware ISO timestamp for inspect")
    parser.add_argument("--input", type=Path, help="JSON facts or decision to save")
    parser.add_argument("--id", help="preregistered candidate or finding identifier")
    parser.add_argument("--artifact", type=Path, help="artifact bytes to hash; never copied to state")
    args = parser.parse_args()
    try:
        if args.action == "init":
            initialize(args.root)
            result = {"initialized": True}
        elif args.action == "record":
            if args.event is None:
                raise ValueError("record requires --event")
            record(args.root, args.event)
            result = {"recorded": True}
        elif args.action in ("inspect", "status"):
            now = datetime.fromisoformat(args.now.replace("Z", "+00:00")) if args.now else datetime.now(timezone.utc)
            result = inspect(args.root, now) if args.action == "inspect" else state_store.status(args.root, now)
        else:
            if args.action not in ("update", "holdout-register", "recover") and args.input is None:
                raise ValueError(f"{args.action} requires --input")
            payload = state_store.read_json(args.input) if args.input else {}
            result = state_store.transact(args.root, args.action, payload, args.id, args.artifact)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
