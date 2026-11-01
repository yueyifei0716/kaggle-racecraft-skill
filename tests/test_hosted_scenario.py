"""URL-to-final-selection acceptance, with no live platform or paid AI calls."""

import csv
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from managed import ManagedRunner
from mandate import MandateStore
from competition_adapter import AdapterError, CompetitionAdapter
import state_store


FIXTURE = ROOT / "tests/fixtures/hosted_worker.py"


def write_csv(path, names, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


class HostedScenario(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.work = self.base / "work"
        self.work.mkdir()
        self.now = datetime.now(timezone.utc)
        self.deadline = self.now + timedelta(days=3)
        a = [-.8, .8, .2, -.2, .6, -.6, .4, -.4, 1., -1., .3, -.3, .9, -.9, .5, -.5]
        b = [.4, -.8, .8, -.3, -.6, .2, -1., .6, -.4, .3, 1., -.2, -.5, .9, -.9, .5]
        rows = []
        for unit in range(16):
            for index, x in enumerate([-2., -1., 1., 2.]):
                rows.append({"row_id": f"r{unit:02}-{index}", "unit": f"u{unit:02}", "fold": str(unit % 3),
                             "group": f"g{unit:02}", "x": x, "a": a[unit], "b": b[unit],
                             "y_true": 3 + 2*x + a[unit]*math.sin(x) + b[unit]*math.cos(x)})
        self.source = self.base / "public_train.csv"
        write_csv(self.source, list(rows[0]), rows)
        inputs = [{"id": str(i), "x": x, "a": a[i], "b": b[i]} for i, x in enumerate([-.7, .6, 1.1, -.8, 1.5, -1.2])]
        self.test_source = self.base / "public_test.csv"
        write_csv(self.test_source, list(inputs[0]), inputs)
        labels = [3+2*row["x"]+row["a"]*math.sin(row["x"])+row["b"]*math.cos(row["x"]) for row in inputs]
        self.mandate_file = self.base / "mandate.json"
        self.mandate_file.write_text(json.dumps({
            "schema_version": 1, "user_confirmed": True, "status": "active",
            "scope": {"competition": "local-acceptance", "url": "https://www.kaggle.com/competitions/local-acceptance"},
            "expires_at": (self.now + timedelta(days=5)).isoformat(),
            "allowed_operations": ["read", "worker", "join", "download", "train", "validate", "upload", "submit", "select_final"],
            "budgets": {"money": 2, "gpu_seconds": 0, "wall_seconds": 400, "submissions": 4, "uploads": 4, "train_runs": 6},
            "max_daily_submissions": 4,
            "rules_acceptance": {"mode": "delegated", "allowed": True,
                                 "constraints": {"no_payment": True, "no_publication": True, "no_personal_disclosure": True}},
            "privacy": {"private_only": True, "no_publication": True, "no_personal_disclosure": True}}))
        self.platform_state = self.base / "platform.json"
        self.config = {
            "root": str(self.work), "competition": "local-acceptance", "mandate_file": str(self.mandate_file),
            "adapter": {"backend": "fake", "state_path": str(self.platform_state),
                        "rules": {"text": "Offline test: private notebooks, no payment or disclosure. Select two finals.", "final_selection_count": 2},
                        "data": {"train_path": str(self.source), "test_path": str(self.test_source), "deadline": self.deadline.isoformat()},
                        "heldout": {"metric": "mae", "labels": labels, "inputs": inputs,
                                    "prediction_command": [sys.executable, str(FIXTURE), "predict"], "prediction_timeout_seconds": 4}},
            "worker_command": [sys.executable, str(FIXTURE), "worker"],
            "worker_timeout_seconds": 4, "training_max_seconds": 4, "validator_timeout_seconds": 4,
            "worker_max_costs": {"wall_seconds": 4, "money": 0, "gpu_seconds": 0},
            "validator_max_costs": {"wall_seconds": 4, "money": 0, "gpu_seconds": 0},
            "command_max_costs": {"wall_seconds": 4, "money": 0, "gpu_seconds": 0},
            "interval_seconds": 1, "intel_interval_seconds": 2, "freeze_hours": .001,
            "max_errors": 1, "max_no_gain": 5, "gate": {"resamples": 100, "min_folds": 3, "min_groups": 2}}

    def run_scenario(self, max_ticks=35):
        inserted_discussion = False
        states = []
        for tick in range(max_ticks):
            # Recreate the orchestrator each time to exercise durable state recovery.
            runner = ManagedRunner(self.config)
            state = runner.tick(self.now + timedelta(seconds=2 * tick))
            states.append(state["status"])
            if state["status"] == "blocked":
                self.fail("hosted flow blocked: " + state.get("reason", "unknown"))
            if self.platform_state.exists() and not inserted_discussion:
                platform = json.loads(self.platform_state.read_text())
                if platform["submissions"]:
                    platform["intel"].append({"id": "comment-1", "kind": "comment", "text": "Try a complementary cosine feature, validate off-fold."})
                    self.platform_state.write_text(json.dumps(platform))
                    inserted_discussion = True
            if state["status"] == "complete":
                return state, states
        self.fail("hosted flow never reached verified final selection: " + str(state))

    def test_single_mandate_bootstrap_train_two_notebooks_and_select_platform(self):
        self.assertFalse((self.work / "competition.json").exists())
        final, states = self.run_scenario()
        platform = json.loads(self.platform_state.read_text())
        self.assertTrue(platform["joined"])
        self.assertEqual(len(platform["selected"]), 2)
        self.assertEqual(len(platform["submissions"]), 2)
        self.assertTrue(all(item["kind"] == "notebook" for item in platform["submissions"].values()))
        self.assertEqual(set(final["final_readback"]["refs"]), set(platform["selected"]))
        ledger = json.loads((self.work / "ledger.json").read_text())
        self.assertEqual(len(ledger["final_selection"]["candidate_ids"]), 2)
        self.assertTrue(ledger["holdout"]["opened_at"])
        self.assertTrue(all(item["gate"]["decision"] == "PASS" for item in ledger["candidates"]))
        requests = [json.loads(line) for line in (self.work / "fixture-worker-requests.jsonl").read_text().splitlines()]
        self.assertEqual(requests[0]["task_kind"], "bootstrap")
        self.assertTrue(any(item["task_kind"] == "reevaluate" and item["intel"].get("events") for item in requests))
        self.assertEqual(sum(item["task_kind"] == "holdout" for item in requests), 1)
        budget = MandateStore(self.work, self.mandate_file).inspect()
        self.assertEqual(budget["remaining"]["submissions"], 2)
        self.assertTrue(all(item["status"] == "complete" for item in budget["receipts"]))
        self.assertTrue(any(item["operation"] == "download" for item in budget["receipts"]))
        self.assertEqual((self.work / "inputs/train.csv").read_bytes(), self.source.read_bytes())
        self.assertFalse(list(self.work.rglob("*approval*.json")))

    def test_insufficient_submission_budget_stops_without_overspend(self):
        mandate = json.loads(self.mandate_file.read_text())
        mandate["budgets"]["submissions"] = 1
        self.mandate_file.write_text(json.dumps(mandate))
        for tick in range(25):
            state = ManagedRunner(self.config).tick(self.now + timedelta(seconds=2*tick))
            if state["status"] == "blocked":
                break
        self.assertEqual(state["status"], "blocked")
        platform = json.loads(self.platform_state.read_text())
        self.assertEqual(len(platform["submissions"]), 1)
        self.assertFalse(platform["selected"])
        self.assertEqual(MandateStore(self.work, self.mandate_file).inspect()["remaining"]["submissions"], 0)

    def test_unknown_submission_reconciles_without_duplicate_submission(self):
        original = CompetitionAdapter._call
        dropped = []
        def interrupt(adapter, action, payload=None, operation_id=None):
            result = original(adapter, action, payload, operation_id)
            if action == "submit" and not dropped:
                dropped.append(operation_id)
                raise AdapterError("synthetic response lost after platform accepted the exact artifact")
            return result
        with patch.object(CompetitionAdapter, "_call", interrupt):
            final, _ = self.run_scenario()
        self.assertEqual(final["status"], "complete")
        platform = json.loads(self.platform_state.read_text())
        self.assertEqual(len(platform["submissions"]), 2)
        budget = MandateStore(self.work, self.mandate_file).inspect()
        self.assertEqual(budget["remaining"]["submissions"], 2)
        self.assertEqual(len([item for item in budget["receipts"] if item["operation"] == "submit"]), 2)
        self.assertTrue(dropped)

    def test_crash_after_holdout_open_recovers_without_second_open_or_retraining(self):
        original = state_store.transact
        crashed = []
        def interrupt(root, action, *args, **kwargs):
            result = original(root, action, *args, **kwargs)
            if action == "holdout-open" and not crashed:
                crashed.append(True)
                raise SystemExit("synthetic controller crash after durable holdout opening")
            return result
        with patch.object(state_store, "transact", interrupt):
            with self.assertRaises(SystemExit):
                self.run_scenario()
        before = MandateStore(self.work, self.mandate_file).inspect()
        trained_before = sum(item["operation"] == "train" for item in before["receipts"])
        final, _ = self.run_scenario(max_ticks=50)
        self.assertEqual(final["status"], "complete")
        ledger = json.loads((self.work / "ledger.json").read_text())
        self.assertEqual(sum(item["event"] == "opened" for item in ledger["holdout"]["audit"]), 1)
        after = MandateStore(self.work, self.mandate_file).inspect()
        self.assertEqual(sum(item["operation"] == "train" for item in after["receipts"]), trained_before)


if __name__ == "__main__":
    unittest.main()
