import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("workspace_state_test", SCRIPTS / "workspace.py")
workspace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workspace)
store = workspace.state_store


class StateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        workspace.initialize(self.root)
        self.config = json.loads((self.root / "competition.json").read_text())
        self.config["metric"] = {"name": "accuracy", "direction": "higher"}
        self.config["gates"] = {name: {"decision": "PASS", "reason": "reviewed"} for name in store.GATES}
        self.write_config()
        self.artifact = self.root / "predictions.csv"
        self.artifact.write_text("id,p\na,0.5\n")

    def write_config(self):
        (self.root / "competition.json").write_text(json.dumps(self.config))

    def save(self, action, payload=None, identifier=None, artifact=None):
        return store.transact(self.root, action, payload, identifier, artifact)

    def ledger(self):
        return store.read_json(self.root / "ledger.json")

    def register(self, identifier="base", parent=None, structural=True):
        return self.save("register", {"id": identifier, "family": "trees", "parent": parent,
                                    "structural": structural, "swept": False,
                                    "hypothesis": "predeclared change improves validation", "stages": ["fit", "predict"]})

    def evidence(self, identifier="base"):
        self.save("update", {}, identifier, self.artifact)
        sha = store.digest(self.artifact)
        self.save("validator", {"authority": "full", "covered_stages": ["fit", "predict"]})
        self.save("score", {"artifact_sha256": sha, "oof": .6, "public": .61,
                            "gate": {"artifact_sha256": sha, "decision": "PASS", "validator_authority": "full",
                                     "covered_stages": ["fit", "predict"], "evidence": {"analysis": "paired comparison", "metric": self.config["metric"]}}}, identifier)

    def trust(self, ids=None):
        self.save("trust", {"regime": "ALIGNED", "candidate_ids": ids or ["base"], "reason": "paired evidence",
                            "computed_at": "2026-11-01T00:00:00Z", "n_paired": 5, "public_units": 100})

    def test_preregistration_rejects_results_duplicates_and_unknown_parent(self):
        with self.assertRaisesRegex(ValueError, "preregistered"):
            self.save("score", {"artifact_sha256": "a" * 64, "oof": .1}, "missing")
        with self.assertRaisesRegex(ValueError, "preregister"):
            self.save("register", {"id": "x", "oof": .1})
        self.register()
        with self.assertRaisesRegex(ValueError, "already registered"):
            self.register()
        with self.assertRaisesRegex(ValueError, "preregistered"):
            self.register("child", "missing")
        self.assertEqual(len(self.ledger()["candidates"]), 1)

    def test_hash_update_invalidates_gate_scores_trust_and_promotion(self):
        self.register()
        with self.assertRaisesRegex(ValueError, "cannot promote"):
            self.save("update", {"status": "promoted"}, "base")
        self.evidence()
        self.save("update", {"status": "promoted"}, "base")
        self.trust()
        self.artifact.write_text("id,p\na,0.7\n")
        self.save("update", {}, "base", self.artifact)
        ledger = self.ledger()
        self.assertEqual(ledger["candidates"][0]["status"], "registered")
        for key in ("gate", "gate_artifact_sha256", "oof", "public"):
            self.assertIsNone(ledger["candidates"][0][key])
        self.assertIsNone(ledger["trust"])
        self.assertIsNone(ledger["final_selection"])

    def test_pipeline_coverage_and_hash_bindings_enforced(self):
        self.register()
        self.evidence()
        self.save("validator", {"authority": "proxy", "covered_stages": ["predict"]})
        with self.assertRaisesRegex(ValueError, "authority|cover"):
            self.save("update", {"status": "promoted"}, "base")
        before = (self.root / "ledger.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "artifact"):
            self.save("score", {"artifact_sha256": "f" * 64, "oof": .8}, "base")
        self.assertEqual(before, (self.root / "ledger.json").read_bytes())
        self.save("update", {"stages": ["fit", "predict", "postprocess"]}, "base")
        self.assertIsNone(self.ledger()["candidates"][0]["gate"])

    def test_findings_resolution_and_final_holdout_decision(self):
        self.register()
        self.evidence()
        self.save("finding", {"id": "leak", "severity": "blocker", "description": "timing leak suspected"})
        with self.assertRaisesRegex(ValueError, "blocker"):
            self.save("final", {"candidate_ids": ["base"]})
        with self.assertRaisesRegex(ValueError, "reason"):
            self.save("resolve", {"status": "accepted"}, "leak")
        self.save("resolve", {"status": "accepted", "reason": "independent review accepted documented residual risk"}, "leak")
        self.save("update", {"status": "promoted"}, "base")
        self.trust()
        with self.assertRaisesRegex(ValueError, "holdout_decision"):
            self.save("final", {"candidate_ids": ["base"], "reason": "chosen from current evidence"})
        self.save("final", {"candidate_ids": ["base"], "reason": "chosen from current evidence",
                            "holdout_decision": {"decision": "not_opened", "reason": "preserve sealed evidence"}})
        self.assertEqual(self.ledger()["final_selection"]["candidate_ids"], ["base"])

    def test_sealed_ids_hash_only_and_open_once(self):
        ids = self.root / "private-unit-ids.txt"
        ids.write_text("sensitive-unit-one\nsensitive-unit-two\n")
        self.save("holdout-register", artifact=ids)
        sha = hashlib.sha256(ids.read_bytes()).hexdigest()
        self.assertEqual(self.ledger()["holdout"]["sha256"], sha)
        self.assertNotIn("sensitive-unit", (self.root / "ledger.json").read_text())
        ids.write_text("modified")
        with self.assertRaisesRegex(ValueError, "match"):
            self.save("holdout-open", {"reason": "one final check"}, artifact=ids)
        self.assertIsNone(self.ledger()["holdout"]["opened_at"])
        ids.write_text("sensitive-unit-one\nsensitive-unit-two\n")
        self.save("holdout-open", {"reason": "one final check"}, artifact=ids)
        before = (self.root / "ledger.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "already opened"):
            self.save("holdout-open", {"reason": "second peek"}, artifact=ids)
        self.assertEqual(before, (self.root / "ledger.json").read_bytes())
        self.assertEqual([item["event"] for item in self.ledger()["holdout"]["audit"]], ["sealed", "opened"])

    def test_status_is_read_only_and_uses_deadline_midpoint_facts(self):
        self.config.update(deadline="2026-11-02T00:00:00Z", started_at="2026-10-20T00:00:00Z", reserve_hours=30)
        self.config["gates"]["midpoint"] = None
        self.write_config()
        before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in self.root.iterdir() if path.is_file()}
        result = store.status(self.root, datetime(2026, 11, 1, tzinfo=timezone.utc))
        self.assertEqual(result["hours_remaining"], 24)
        for value in ("freeze_and_finalize", "midpoint_review_due", "late_join_build_minimal_valid_baseline", "build_baseline"):
            self.assertIn(value, result["recommendations"])
        self.assertEqual(before, {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in self.root.iterdir() if path.is_file()})

    def test_legacy_refusal_does_not_change_work(self):
        legacy = self.root / "old"
        legacy.mkdir()
        path = legacy / "GAMEPLAN.md"
        path.write_text("manual historical evidence\n")
        result = store.status(legacy, datetime.now(timezone.utc))
        self.assertEqual(result["state"], "legacy_not_migrated")
        with self.assertRaisesRegex(ValueError, "legacy_not_migrated"):
            workspace.initialize(legacy)
        self.assertEqual(path.read_text(), "manual historical evidence\n")
        self.assertFalse((legacy / "competition.json").exists())

    def test_failed_atomic_write_preserves_state_and_next_write_recovers(self):
        before = (self.root / "ledger.json").read_bytes()
        with patch.object(store.os, "replace", side_effect=OSError("simulated disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                self.register()
        self.assertEqual(before, (self.root / "ledger.json").read_bytes())
        self.assertEqual(list(self.root.glob(".ledger.json.*")), [])
        self.register()
        self.assertEqual(self.ledger()["candidates"][0]["id"], "base")

    def test_cli_concurrent_register_and_event_append_have_no_lost_updates(self):
        jobs = []
        for index in range(10):
            path = self.root / f"input-{index}.json"
            path.write_text(json.dumps({"id": f"run-{index}", "family": "trees", "structural": False,
                                        "swept": False, "hypothesis": "predeclared", "stages": ["fit"]}))
            jobs.append(subprocess.Popen([sys.executable, str(SCRIPTS / "workspace.py"), "register", str(self.root), "--input", str(path)],
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        for job in jobs:
            stdout, stderr = job.communicate(timeout=30)
            self.assertEqual(job.returncode, 0, stderr)
            self.assertIn("registered", json.loads(stdout))
        self.assertEqual(len(self.ledger()["candidates"]), 10)
        jobs = []
        for index in range(10):
            path = self.root / f"event-{index}.json"
            path.write_text(json.dumps({"id": f"event-{index}", "event": "planned"}))
            jobs.append(subprocess.Popen([sys.executable, str(SCRIPTS / "workspace.py"), "record", str(self.root), "--event", str(path)],
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        for job in jobs:
            _, stderr = job.communicate(timeout=30)
            self.assertEqual(job.returncode, 0, stderr)
        self.assertEqual(workspace.inspect(self.root, datetime.now(timezone.utc))["event_count"], 10)

    def test_cli_invalid_input_preserves_state_and_reports_error(self):
        path = self.root / "bad.json"
        path.write_text('{"id":"future", "oof":NaN}')
        before = (self.root / "ledger.json").read_bytes()
        result = subprocess.run([sys.executable, str(SCRIPTS / "workspace.py"), "register", str(self.root), "--input", str(path)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("non-finite", result.stderr)
        self.assertEqual(before, (self.root / "ledger.json").read_bytes())

    def test_structural_pairs_trigger_stale_trust_and_config_changes_stale_it(self):
        self.register(structural=False)
        self.evidence()
        ids = ["base"]
        for index in range(5):
            identifier = f"struct-{index}"
            self.register(identifier, "base")
            self.evidence(identifier)
            ids.append(identifier)
        result = store.status(self.root, datetime.now(timezone.utc))
        self.assertEqual(result["structural_pairs"], 5)
        self.assertIn("refresh_cv_public_trust", result["recommendations"])
        self.trust(ids)
        self.assertTrue(store.status(self.root, datetime.now(timezone.utc))["trust_current"])
        self.config["public_units"] = 101
        self.write_config()
        self.assertFalse(store.status(self.root, datetime.now(timezone.utc))["trust_current"])

    def test_config_syncs_metric_public_units_and_recovers_interrupted_pair(self):
        self.save("config", {"metric": {"name": "rmse", "direction": "lower"}, "public_units": 400})
        self.assertEqual(self.ledger()["metric"], {"name": "rmse", "direction": "lower"})
        self.assertEqual(self.ledger()["public_units"], 400)
        original_replace = store.os.replace

        def fail_ledger(source, target):
            if Path(target).name == "ledger.json":
                raise OSError("simulated ledger write failure")
            return original_replace(source, target)

        with patch.object(store.os, "replace", side_effect=fail_ledger):
            with self.assertRaisesRegex(OSError, "ledger write failure"):
                self.save("config", {"public_units": 500})
        before = (self.root / ".state-transaction.json").read_bytes()
        self.assertEqual(store.status(self.root, datetime.now(timezone.utc))["state"], "recovery_required")
        self.assertEqual(before, (self.root / ".state-transaction.json").read_bytes())
        self.assertEqual(self.save("recover"), {"recovered": True})
        self.assertEqual(self.ledger()["public_units"], 500)
        self.assertFalse((self.root / ".state-transaction.json").exists())

    def test_selection_unit_schema_sweeps_and_local_only_insufficient_final(self):
        self.register()
        self.evidence()
        self.save("score", {"artifact_sha256": store.digest(self.artifact), "selection_units": {"unit-a": .5, "unit-b": .6}}, "base")
        self.assertEqual(self.ledger()["candidates"][0]["selection_units"], {"unit-a": .5, "unit-b": .6})
        self.assertEqual(self.ledger()["candidates"][0]["swept"], 1)
        with self.assertRaisesRegex(ValueError, "selection_units"):
            self.save("update", {"selection_units": 2}, "base")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.save("score", {"artifact_sha256": store.digest(self.artifact), "oof": {"score": .6}}, "base")
        self.save("update", {"status": "promoted"}, "base")
        self.save("trust", {"regime": "INSUFFICIENT", "candidate_ids": ["base"], "reason": "too few public units; rely on local OOF"})
        self.save("final", {"candidate_ids": ["base"], "reason": "local OOF choice", "holdout_decision": {"decision": "not_used", "reason": "small data"}})
        self.assertEqual(self.ledger()["final_selection"]["evidence_basis"], "local_only")

    def test_independent_structural_roots_and_critical_blockers(self):
        for index in range(5):
            self.register(f"root-{index}")
            self.evidence(f"root-{index}")
        self.assertIn("refresh_cv_public_trust", store.status(self.root, datetime.now(timezone.utc))["recommendations"])
        self.save("finding", {"id": "critical", "severity": "critical", "description": "invalid validation"})
        with self.assertRaisesRegex(ValueError, "blocker"):
            self.save("update", {"status": "promoted"}, "root-0")

    def test_sweep_cannot_understate_trials_or_skip_fresh_replication(self):
        self.save("register", {"id": "sweep", "family": "trees", "structural": False, "swept": 5,
                               "hypothesis": "predeclared search", "stages": ["fit", "predict"]})
        self.evidence("sweep")
        with self.assertRaisesRegex(ValueError, "search_trials|replication"):
            self.save("update", {"status": "promoted"}, "sweep")
        sha = store.digest(self.artifact)
        gate = {"decision": "PASS", "artifact_sha256": sha, "validator_authority": "full",
                "covered_stages": ["fit", "predict"], "evidence": {"search_trials": 5, "metric": self.config["metric"],
                "replication": {"fresh": True, "independent": True, "candidate_sha256": sha, "evidence_id": "independent-run"}}}
        self.save("score", {"artifact_sha256": sha, "gate": gate}, "sweep")
        self.save("update", {"status": "promoted"}, "sweep")
        self.assertEqual(self.ledger()["candidates"][0]["status"], "promoted")

    def test_final_requires_returned_public_score(self):
        self.register()
        self.evidence()
        ledger = self.ledger()
        ledger["candidates"][0]["public"] = None
        store.atomic_json(self.root / "ledger.json", ledger)
        self.save("update", {"status": "promoted"}, "base")
        self.trust()
        with self.assertRaisesRegex(ValueError, "public score"):
            self.save("final", {"candidate_ids": ["base"], "reason": "local choice",
                                "holdout_decision": {"decision": "not_used", "reason": "small data"}})

    def test_gate_metric_missing_name_or_direction_mismatch_blocks_promotion(self):
        self.register()
        self.evidence()
        current = self.ledger()["candidates"][0]
        sha = current["artifact_sha256"]
        for metric in (None, {"name": "rmse", "direction": "higher"}, {"name": "accuracy", "direction": "lower"}):
            gate = dict(current["gate"], evidence={"metric": metric})
            self.save("score", {"artifact_sha256": sha, "gate": gate}, "base")
            with self.assertRaisesRegex(ValueError, "gate metric"):
                self.save("update", {"status": "promoted"}, "base")
        self.save("score", {"artifact_sha256": sha, "gate": current["gate"]}, "base")
        self.save("update", {"status": "promoted"}, "base")
        self.assertEqual(self.ledger()["candidates"][0]["status"], "promoted")


if __name__ == "__main__":
    unittest.main()
