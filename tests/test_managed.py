import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import managed


class Store:
    def __init__(self, allowed=True):
        self.allowed = allowed
        self.receipts = []

    def inspect(self):
        return {"status": "active", "expired": False, "remaining": {"gpu_seconds": 0},
                "mandate": {"scope": {"competition": "local", "url": "https://www.kaggle.com/competitions/local"},
                            "rules_acceptance": {"allowed": self.allowed, "mode": "pinned", "document_sha256": "a" * 64}},
                "receipts": self.receipts}

    def reserve(self, operation, competition, identity, costs, metadata=None):
        receipt = {"id": str(len(self.receipts)), "operation": operation, "identity": identity,
                   "status": "pending", "reserved_costs": costs}
        self.receipts.append(receipt)
        return receipt

    def complete(self, identifier, result):
        self.receipts[int(identifier)].update(status="complete", result=result)

    def fail(self, identifier, reason, unknown=True):
        self.receipts[int(identifier)].update(status="unknown" if unknown else "failed", reason=reason)


class Adapter:
    def __init__(self):
        self.joins = 0

    def inspect(self):
        return {"rules": {"sha256": "a" * 64, "current": True, "final_selection_count": 1},
                "joined": True, "joined_rules_hash": "a" * 64}

    def join(self, sha):
        self.joins += 1
        return {"status": "succeeded"}

    def refresh_intel(self, cursor):
        return {"events": [], "cursor": "1"}


class ManagedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = {"root": str(self.root), "competition": "local", "mandate_file": "unused",
                       "worker_command": [sys.executable, "-c", "print('{}')"], "worker_timeout_seconds": 1,
                       "training_max_seconds": 1, "validator_timeout_seconds": 1, "interval_seconds": 1}
        self.store, self.adapter = Store(), Adapter()
        self.runner = managed.ManagedRunner(self.config, self.adapter, self.store)

    def tearDown(self):
        self.temp.cleanup()

    def test_strict_worker_json_rejects_duplicate_nonfinite_and_prose(self):
        for value in ('{"x":1,"x":2}', '{"x":NaN}', '```json\n{}\n```', '[]'):
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                managed.strict_json(value)

    def test_local_process_reserves_before_execution_and_preserves_unknown_budget(self):
        output = self.root / "started"
        code = "from pathlib import Path; import time; Path('started').write_text('yes'); time.sleep(10)"
        with self.assertRaises(subprocess.TimeoutExpired):
            self.runner._budgeted("train", "test", {}, [sys.executable, "-c", code], {}, 1)
        self.assertTrue(output.exists())
        self.assertEqual(self.store.receipts[0]["status"], "unknown")
        self.assertEqual(self.store.receipts[0]["reserved_costs"]["wall_seconds"], 1)
        self.assertEqual(self.store.receipts[0]["reserved_costs"]["train_runs"], 1)

    def test_timeout_kills_entire_child_process_group(self):
        child_pid = self.root / "child.pid"
        code = "import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c',\"import time; from pathlib import Path; time.sleep(.4); Path('survived').write_text('yes')\"]); Path('child.pid').write_text(str(p.pid)); time.sleep(60)"
        with self.assertRaises(subprocess.TimeoutExpired):
            managed.command([sys.executable, "-c", code], self.root, {}, .2)
        self.assertGreater(int(child_pid.read_text()), 0)
        time.sleep(.4)
        self.assertFalse((self.root / "survived").exists())

    def test_rules_not_approved_blocks_before_any_worker(self):
        self.store.allowed = False
        result = self.runner.tick()
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(self.store.receipts, [])
        self.assertEqual(self.adapter.joins, 0)

    def test_persisted_due_time_survives_new_runner(self):
        now = datetime.now(timezone.utc)
        self.runner.state["next_due_at"] = (now + timedelta(hours=1)).isoformat()
        self.runner._save()
        runner = managed.ManagedRunner(self.config, self.adapter, self.store)
        self.assertEqual(runner.tick(now)["attempt"], 0)
        self.assertEqual(self.store.receipts, [])

    def test_worker_cannot_overwrite_manual_file_or_escape_root(self):
        (self.root / "manual.py").write_text("manual")
        for filename in ("manual.py", "../escape.py", "ledger.json"):
            if filename == "ledger.json":
                (self.root / filename).write_text("facts")
            with self.assertRaises(managed.ManagedError):
                self.runner._files([{"path": filename, "content": "changed"}])
        self.assertEqual((self.root / "manual.py").read_text(), "manual")

    def test_worker_unknown_fields_are_not_silently_ignored(self):
        with self.assertRaises(managed.ManagedError):
            self.runner._schema({"kind": "experiment", "reason": "x", "gate": "PASS"})

    def test_default_agent_request_preserves_user_goal_and_authorized_url(self):
        config = {**self.config, "goal": "研究优先，争取铜牌", "worker_max_costs": {"money": 0}}
        config.pop("worker_command")
        self.runner.state["setup"] = {"goal": "worker 替换的目标"}
        self.runner._save()
        runner = managed.ManagedRunner(config, self.adapter, self.store)
        with patch.object(runner, "_budgeted", return_value={"kind": "stop", "reason": "test"}) as budgeted:
            runner._worker("bootstrap", managed.instant())
        request = budgeted.call_args.args[4]
        self.assertEqual(request["goal"], config["goal"])
        self.assertEqual(request["objective"], config["goal"])
        self.assertEqual(request["competition_url"], self.store.inspect()["mandate"]["scope"]["url"])
        self.assertEqual(request["platform"]["url"], request["competition_url"])
        self.assertIn("agent_process.py", " ".join(budgeted.call_args.args[3]))

    def test_prediction_truth_identity_is_verified(self):
        truth = self.root / "truth.csv"
        truth.write_text("row_id,unit,fold,y_true\na,u,f,1\n")
        rows = self.root / "rows.csv"
        rows.write_text("row_id,unit,fold,y_true,y_pred\na,u,f,2,2\n")
        with self.assertRaisesRegex(managed.ManagedError, "truth"):
            self.runner._rows(rows, truth)

    def test_runner_lock_blocks_second_instance(self):
        with self.runner._lock():
            other = managed.ManagedRunner(self.config, self.adapter, self.store)
            with self.assertRaisesRegex(managed.ManagedError, "another managed runner"):
                other.tick()

    def test_unresolved_training_never_reexecutes(self):
        self.runner.state.update(phase="iterate", active_plan={"registration": {"id": "a"}})
        self.runner._save()
        result = self.runner.tick()
        self.assertIn("interrupted training", result["reason"])
        self.assertEqual(self.store.receipts, [])

    def test_completed_training_resumes_evaluation_without_another_process(self):
        plan = {"registration": {"id": "a"}}
        self.store.receipts.append({"id": "0", "status": "complete", "result": {}})
        self.runner.state.update(phase="iterate", active_plan=plan,
                                 jobs=[{"id": "0", "identity": {"task": "experiment", "attempt": 0}}])
        self.runner._save()
        with patch.object(self.runner, "_evaluate") as evaluate:
            self.runner.tick()
        self.assertEqual(evaluate.call_args.args[0], plan)
        self.assertEqual(len(self.store.receipts), 1)

    def test_atomic_supervisor_receipt_recovers_interrupted_parent(self):
        reservation = self.store.reserve("train", "local", "job", {"train_runs": 1, "wall_seconds": 2})
        folder = self.root / ".kaggle" / "jobs"
        folder.mkdir(parents=True)
        result = folder / "0.result.json"
        process = folder / "0.process.json"
        request = folder / "0.request.json"
        request.write_text(json.dumps({"operation_id": "0", "argv": [sys.executable, "-c", "print('{\"trained\":true}')"],
                                       "root": str(self.root), "payload": {}, "timeout": 2,
                                       "result_path": str(result), "process_path": str(process)}))
        self.runner.state["jobs"] = [{"id": "0", "identity": "job", "status": "running", "result_path": str(result)}]
        self.runner._save()
        managed.command([sys.executable, str(Path(managed.__file__).resolve()), "_job", str(request)], self.root, {}, 2)
        self.assertEqual(self.store.receipts[0]["status"], "pending")
        restored = managed.ManagedRunner(self.config, self.adapter, self.store)
        restored._recover_jobs()
        self.assertEqual(self.store.receipts[0]["status"], "complete")
        self.assertEqual(self.store.receipts[0]["result"], {"trained": True})

    def test_budget_denial_prevents_process_start(self):
        def deny(*args, **kwargs):
            raise ValueError("budget exhausted")
        self.store.reserve = deny
        with self.assertRaisesRegex(ValueError, "budget exhausted"):
            self.runner._budgeted("train", "test", {}, [sys.executable, "-c", "open('bad','w').write('x')"], {}, 1)
        self.assertFalse((self.root / "bad").exists())

    def test_short_bounded_serve_returns_without_cross_session_chat(self):
        self.runner.state["next_due_at"] = (managed.instant() + timedelta(seconds=10)).isoformat()
        self.runner._save()
        start = time.monotonic()
        self.runner.serve(max_seconds=.05)
        self.assertLess(time.monotonic() - start, 1)

    def test_temporary_pid_service_starts_prevents_duplicate_and_uninstalls(self):
        mandate_path = self.root / "mandate.json"
        mandate_path.write_text(json.dumps({
            "schema_version": 1, "user_confirmed": True,
            "scope": {"competition": "local", "url": "https://www.kaggle.com/competitions/local"},
            "expires_at": (managed.instant() + timedelta(days=1)).isoformat(),
            "allowed_operations": ["read"],
            "budgets": {"money": 0, "gpu_seconds": 0, "wall_seconds": 0, "submissions": 0, "uploads": 0, "train_runs": 0},
            "max_daily_submissions": 0,
            "rules_acceptance": {"mode": "pinned", "allowed": False, "document_sha256": None},
            "privacy": {"private_only": True, "no_publication": True, "no_personal_disclosure": True}}))
        config = {**self.config, "mandate_file": str(mandate_path), "worker_max_costs": {"money": 0},
                  "adapter": {"backend": "fake", "state_path": str(self.root / "platform.json"),
                              "rules": {"text": "offline test", "final_selection_count": 1}}}
        config_path = self.root / "managed.json"
        config_path.write_text(json.dumps(config))
        processes = []
        real_popen = subprocess.Popen
        def spawn(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process
        with patch.object(managed.subprocess, "Popen", side_effect=spawn):
            service = managed.install(config_path)
        try:
            with self.assertRaisesRegex(managed.ManagedError, "already running"):
                managed.install(config_path)
            self.assertEqual(processes[0].wait(timeout=3), 0)
            self.assertTrue(managed.uninstall(self.root)["already_stopped"])
            self.assertFalse((self.root / "managed-service.json").exists())
        finally:
            try:
                os.kill(service["pid"], 9)
            except ProcessLookupError:
                pass


if __name__ == "__main__":
    unittest.main()
