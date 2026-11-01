"""Entirely offline adapter tests; no credentials or platform clients are consulted."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import time
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from competition_adapter import AdapterError, CompetitionAdapter, PROTOCOL, validate_candidate


class FakeMandate:
    def __init__(self, path):
        self.path = path
        if not path.exists():
            path.write_text("[]")

    def inspect(self):
        return {"receipts": json.loads(self.path.read_text())}

    def _write(self, receipts):
        self.path.write_text(json.dumps(receipts))

    def reserve(self, operation, competition, identity, costs, metadata=None, now=None):
        records = self.inspect()["receipts"]
        found = next((r for r in records if r["identity"] == identity), None)
        if found:
            return {**found, "duplicate": True}
        receipt = {"id": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(), "operation": operation,
                   "competition": competition, "identity": identity, "costs": costs,
                   "metadata": metadata, "status": "pending"}
        records.append(receipt)
        self._write(records)
        return receipt

    def complete(self, operation_id, result, actual_costs=None):
        records = self.inspect()["receipts"]
        receipt = next(r for r in records if r["id"] == operation_id)
        receipt.update(status="complete", result=result)
        self._write(records)

    def fail(self, operation_id, reason, unknown=True):
        records = self.inspect()["receipts"]
        receipt = next(r for r in records if r["id"] == operation_id)
        receipt.update(status="unknown" if unknown else "failed", reason=reason)
        self._write(records)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = FakeMandate(self.root / "mandate.json")
        self.config = {"backend": "fake", "competition": "synthetic", "state_path": str(self.root / "platform.json"),
                       "rules": {"text": "Synthetic no-fee private competition", "final_selection_count": 3},
                       "heldout": {"labels": [0, 1], "metric": "accuracy"}}
        self.adapter = CompetitionAdapter(self.config, self.store)

    def candidate(self, index=0):
        path = self.root / f"candidate-{index}.csv"
        path.write_text(f"id,prediction\na{index},0.1\nb{index},0.9\n")
        return {"id": f"c{index}", "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "private": True, "kind": "file"}

    def join(self):
        info = self.adapter.inspect()
        result = self.adapter.join(info["rules"]["hash"])
        self.assertEqual(result["status"], "succeeded")
        return result

    def test_persistent_full_loop_scores_and_rule_based_three_final_refs(self):
        self.join()
        refs = []
        for index in range(3):
            candidate = self.candidate(index)
            upload = self.adapter.upload(candidate)
            self.assertTrue(upload["private"])
            submission = self.adapter.submit({**candidate, "upload_ref": upload["upload_ref"]})
            score = self.adapter.score(submission)
            self.assertEqual(score["score"], 1.0)
            self.assertEqual(score["candidate_id"], candidate["id"])
            self.assertEqual(score["artifact_sha256"], candidate["sha256"])
            refs.append(submission["submission_ref"])
        self.adapter.select_final(refs)
        restarted = CompetitionAdapter(self.config, FakeMandate(self.root / "mandate.json"))
        self.assertEqual(restarted.final_selection()["refs"], refs)
        self.assertEqual(len(self.store.inspect()["receipts"]), 8)
        with self.assertRaisesRegex(AdapterError, "selection count"):
            self.adapter.select_final(refs[:2])

    def test_join_reviews_current_document_and_blocks_changed_hash(self):
        self.assertEqual(self.adapter.join("stale")["status"], "blocked")
        self.assertEqual(self.store.inspect()["receipts"], [])
        self.join()
        receipt = self.store.inspect()["receipts"][0]
        self.assertTrue(receipt["metadata"]["rules_review"]["complies"])
        state = json.loads((self.root / "platform.json").read_text())
        state["rules"].update(text="new rules", payment_required=True)
        state["rules"].pop("hash")
        state["rules"].pop("sha256")
        (self.root / "platform.json").write_text(json.dumps(state))
        self.assertEqual(self.adapter.upload(self.candidate())["status"], "blocked")
        self.assertEqual(self.adapter.join(self.adapter.inspect()["rules"]["hash"])["status"], "blocked")
        self.assertEqual(len(self.store.inspect()["receipts"]), 1)

    def test_incremental_topics_comments_are_read_again_after_restart(self):
        self.adapter.inspect()
        state_path = self.root / "platform.json"
        state = json.loads(state_path.read_text())
        state["intel"] = [{"kind": "topic", "id": "t1", "text": "new topic"}]
        state_path.write_text(json.dumps(state))
        first = self.adapter.refresh_intel()
        self.assertEqual(len(first["events"]), 1)
        state["intel"].append({"kind": "comment", "id": "c1", "topic_id": "t1", "text": "new reply"})
        state_path.write_text(json.dumps(state))
        second = CompetitionAdapter(self.config, self.store).refresh_intel(first["cursor"])
        self.assertEqual([event["kind"] for event in second["events"]], ["comment"])
        self.assertEqual(self.adapter.refresh_intel(second["cursor"])["events"], [])

    def test_lost_ack_reconciles_exact_operation_without_duplicate_submission(self):
        self.join()
        original = self.adapter._call
        def lost_ack(action, payload=None, operation_id=None):
            result = original(action, payload, operation_id)
            if action == "submit":
                raise AdapterError("ack lost")
            return result
        with patch.object(self.adapter, "_call", side_effect=lost_ack):
            result = self.adapter.submit(self.candidate())
        self.assertEqual(result["status"], "unknown")
        restarted = CompetitionAdapter(self.config, self.store)
        with patch.object(restarted, "_fake_call", wraps=restarted._fake_call) as call:
            duplicate = restarted.submit(self.candidate())
            self.assertEqual(duplicate["status"], "unknown")
            self.assertFalse(any(c.args[0] == "submit" for c in call.call_args_list))
        confirmed = restarted.reconcile(result["operation_id"])
        self.assertEqual(confirmed["status"], "succeeded")
        state = json.loads((self.root / "platform.json").read_text())
        self.assertEqual(len(state["submissions"]), 1)
        self.assertEqual(restarted.submit(self.candidate()), confirmed)

    def test_reserve_precedes_mutation_and_budget_rejection_has_no_effect(self):
        self.join()
        with patch.object(self.store, "reserve", side_effect=ValueError("budget exhausted")):
            with self.assertRaisesRegex(ValueError, "budget exhausted"):
                self.adapter.submit(self.candidate())
        state = json.loads((self.root / "platform.json").read_text())
        self.assertEqual(state["submissions"], {})

    def test_private_and_actual_hash_are_checked_before_reservation(self):
        self.join()
        candidate = self.candidate()
        with self.assertRaisesRegex(AdapterError, "private"):
            self.adapter.upload({**candidate, "private": False})
        Path(candidate["path"]).write_text("changed")
        with self.assertRaisesRegex(AdapterError, "SHA256"):
            self.adapter.submit(candidate)
        self.assertEqual(len(self.store.inspect()["receipts"]), 1)

    def test_configured_heldout_metric_is_calculated_on_platform(self):
        self.config["heldout"] = {"labels": [0, 1], "metric": "mse"}
        self.adapter = CompetitionAdapter(self.config, self.store)
        self.join()
        result = self.adapter.score(self.adapter.submit(self.candidate()))
        self.assertAlmostEqual(result["score"], .01)
        with self.assertRaisesRegex(AdapterError, "not bound"):
            self.adapter.score("invented-submission")

    def test_unknown_backend_and_absent_bridge_fail_closed(self):
        with self.assertRaisesRegex(AdapterError, "unknown adapter backend"):
            CompetitionAdapter({**self.config, "backend": "kaggle-magic"}, self.store)
        adapter = CompetitionAdapter({"competition": "synthetic", "backend": "command"}, self.store)
        self.assertEqual(adapter.inspect()["status"], "blocked")
        self.assertEqual(adapter.join("hash")["status"], "blocked")
        self.assertEqual(self.store.inspect()["receipts"], [])

    def test_command_protocol_argv_and_output_redaction(self):
        command = ["configured-bridge", "literal-$(no-shell)"]
        adapter = CompetitionAdapter({"competition": "synthetic", "backend": "command", "adapter_command": command,
                                      "timeout_seconds": 2, "bridge_max_costs": {"money": 0, "gpu_seconds": 0, "wall_seconds": 2}}, self.store)
        info = self.adapter.inspect()
        envelope = {"protocol": PROTOCOL, "action": "inspect", "ok": True, "result": info}
        with patch.object(adapter, "_run_bridge", return_value=json.dumps(envelope)) as run:
            self.assertEqual(adapter.inspect()["rules"]["hash"], info["rules"]["hash"])
        self.assertEqual(run.call_args.args[0], command)
        request = run.call_args.args[1]
        self.assertEqual(request["competition"], "synthetic")
        self.assertEqual(request["protocol"], PROTOCOL)
        with patch.object(adapter, "_run_bridge", side_effect=AdapterError("bridge failed; raw process output withheld")):
            with self.assertRaises(AdapterError) as error:
                adapter.inspect()
            self.assertNotIn("SECRET", str(error.exception))

    def test_bridge_capability_missing_is_specific_block_before_reserve(self):
        info = self.adapter.inspect()
        info["capabilities"]["join"] = False
        adapter = CompetitionAdapter({"competition": "synthetic", "backend": "command", "adapter_command": ["bridge"],
                                      "timeout_seconds": 2, "bridge_max_costs": {"money": 0, "gpu_seconds": 0, "wall_seconds": 2}}, self.store)
        response = {"protocol": PROTOCOL, "action": "inspect", "ok": True, "result": info}
        with patch.object(adapter, "_run_bridge", return_value=json.dumps(response)):
            result = adapter.join(info["rules"]["hash"])
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["capability"], "join")
        self.assertTrue(all(receipt["operation"] == "worker" for receipt in self.store.inspect()["receipts"]))

    def command_adapter(self, *, money=10, wall_seconds=20, timeout=.2, maximum_money=.25, body=None):
        from mandate import MandateStore
        mandate_path = self.root / "command-authorization.json"
        mandate_path.write_text(json.dumps({"schema_version": 1, "user_confirmed": True,
            "scope": {"competition": "synthetic", "url": "https://www.kaggle.com/competitions/synthetic"},
            "expires_at": "2030-01-01T00:00:00Z", "allowed_operations": ["worker", "join", "upload", "submit", "select_final", "download"],
            "budgets": {"money": money, "gpu_seconds": 0, "wall_seconds": wall_seconds,
                        "submissions": 3, "uploads": 3, "train_runs": 0},
            "max_daily_submissions": 3,
            "rules_acceptance": {"mode": "delegated", "allowed": True, "constraints": {
                "no_payment": True, "no_publication": True, "no_personal_disclosure": True}},
            "privacy": {"private_only": True, "no_publication": True, "no_personal_disclosure": True}}))
        store = MandateStore(self.root, mandate_path)
        script = self.root / "offline-bridge.py"
        marker = self.root / "bridge-starts.txt"
        info = self.adapter.inspect()
        script.write_text("import json,sys\nfrom pathlib import Path\n"
                          "request=json.load(sys.stdin)\n"
                          + f"marker=Path({str(marker)!r})\n"
                          + "marker.write_text((marker.read_text() if marker.exists() else '')+'started\\n')\n"
                          + (body if body is not None else
                             f"print(json.dumps({{'protocol':{PROTOCOL!r},'action':request['action'],'ok':True,'result':{info!r}}}))\n"))
        config = {"backend": "command", "competition": "synthetic", "adapter_command": [sys.executable, str(script), "literal-$(no-shell)"],
                  "timeout_seconds": timeout, "bridge_max_costs": {"money": maximum_money, "gpu_seconds": 0, "wall_seconds": 1}}
        return CompetitionAdapter(config, store), marker

    def test_command_bridge_zero_budget_rejects_before_executable_starts(self):
        adapter, marker = self.command_adapter(money=0)
        with self.assertRaisesRegex(ValueError, "budget exhausted: money"):
            adapter.inspect()
        self.assertFalse(marker.exists())
        self.assertEqual(adapter.store.inspect()["receipts"], [])

    def test_command_bridge_every_read_charged_and_restart_uses_new_identity(self):
        adapter, marker = self.command_adapter()
        original_popen = subprocess.Popen
        with patch("competition_adapter.subprocess.Popen", wraps=original_popen) as popen:
            adapter.inspect()
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertFalse(popen.call_args.kwargs["shell"])
        self.assertEqual(popen.call_args.args[0][-1], "literal-$(no-shell)")
        adapter.inspect()
        CompetitionAdapter(adapter.config, adapter.store).inspect()
        state = adapter.store.inspect()
        self.assertEqual(marker.read_text().splitlines(), ["started"] * 3)
        self.assertEqual(state["used"]["money"], .75)
        self.assertEqual(state["used"]["wall_seconds"], 3)
        self.assertEqual(len({receipt["id"] for receipt in state["receipts"]}), 3)
        self.assertTrue(all(receipt["status"] == "complete" for receipt in state["receipts"]))

    def test_command_bridge_timeout_charges_unknown_and_kills_descendant_group(self):
        late_marker = self.root / "late-child-write.txt"
        child_code = ("import signal,time;from pathlib import Path;"
                      "signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(.9);"
                      + f"Path({str(late_marker)!r}).write_text('survived')")
        body = ("import signal,subprocess,time\n"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                + f"subprocess.Popen([sys.executable,'-c',{child_code!r}])\n"
                + "time.sleep(10)\n")
        adapter, marker = self.command_adapter(timeout=.2, body=body)
        with self.assertRaisesRegex(AdapterError, "outcome may be unknown"):
            adapter.inspect()
        self.assertTrue(marker.exists())
        state = adapter.store.inspect()
        self.assertEqual(state["receipts"][0]["status"], "unknown")
        self.assertEqual(state["used"]["money"], .25)
        self.assertEqual(state["used"]["wall_seconds"], 1)
        time.sleep(.4)
        self.assertFalse(late_marker.exists(), "timed out bridge left a descendant executing")

    def test_command_bridge_mutation_invocation_never_repeats_after_unknown(self):
        adapter, marker = self.command_adapter(body="print('not-json')\n")
        with self.assertRaisesRegex(AdapterError, "invalid JSON"):
            adapter._call("submit", {}, "a" * 64)
        with self.assertRaisesRegex(AdapterError, "unresolved"):
            CompetitionAdapter(adapter.config, adapter.store)._call("submit", {}, "a" * 64)
        self.assertEqual(marker.read_text().splitlines(), ["started"])
        self.assertEqual(len(adapter.store.inspect()["receipts"]), 1)

    def test_command_bridge_requires_explicit_bounded_maximum_costs(self):
        config = {"backend": "command", "competition": "synthetic", "adapter_command": ["never-executed"]}
        with self.assertRaisesRegex(AdapterError, "explicitly declare"):
            CompetitionAdapter(config, self.store)
        with self.assertRaisesRegex(AdapterError, "cover its timeout"):
            CompetitionAdapter({**config, "bridge_max_costs": {"money": 0, "gpu_seconds": 0, "wall_seconds": 1}}, self.store)

    def test_snapshot_prevents_original_path_race_and_ignores_claimed_predictions(self):
        self.join()
        candidate = self.candidate()
        original = self.adapter._call
        def change_original(action, payload=None, operation_id=None):
            if action == "submit":
                self.assertNotEqual(payload["path"], candidate["path"])
                Path(candidate["path"]).write_text("id,prediction\na,0.9\nb,0.1\n")
            return original(action, payload, operation_id)
        with patch.object(self.adapter, "_call", side_effect=change_original):
            result = self.adapter.submit({**candidate, "predictions": [0.9, 0.1]})
        self.assertEqual(self.adapter.score(result)["score"], 1.0)

    def test_notebook_inference_driver_scores_actual_artifact_and_reserves_time(self):
        driver = self.root / "synthetic_predictor.py"
        driver.write_text("import json,sys\nfrom pathlib import Path\n"
                          "request=json.load(sys.stdin)\n"
                          "notebook=json.loads(Path(request['artifact_path']).read_text())\n"
                          "namespace={}\n"
                          "for cell in notebook['cells']:\n"
                          " if cell['cell_type']=='code': exec(''.join(cell['source']),namespace)\n"
                          "print(json.dumps({'predictions':[namespace['predict'](x) for x in request['inputs']]}))\n")
        self.config["heldout"] = {"labels": [0, 1], "inputs": [1, 3], "metric": "accuracy",
                                  "prediction_command": [sys.executable, str(driver)], "prediction_timeout_seconds": 2}
        self.adapter = CompetitionAdapter(self.config, self.store)
        self.join()
        path = self.root / "inference.ipynb"
        path.write_text(json.dumps({"nbformat": 4, "cells": [
            {"cell_type": "code", "source": ["def predict(x):\n    return float(x >= 2)\n"]}]}))
        candidate = {"id": "notebook", "kind": "notebook", "private": True, "path": str(path),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        submission = self.adapter.submit(candidate)
        self.assertEqual(submission["status"], "succeeded")
        self.assertEqual(self.adapter.score(submission)["score"], 1.0)
        receipt = self.store.inspect()["receipts"][-1]
        self.assertEqual(receipt["costs"]["wall_seconds"], 2)
        self.assertEqual(receipt["costs"]["submissions"], 1)

    def test_real_mandate_store_compatible_with_delegated_rules_and_dedup(self):
        from mandate import MandateStore
        mandate_path = self.root / "authorization.json"
        mandate_path.write_text(json.dumps({"schema_version": 1, "user_confirmed": True,
            "scope": {"competition": "synthetic", "url": "https://www.kaggle.com/competitions/synthetic"},
            "expires_at": "2030-01-01T00:00:00Z", "allowed_operations": ["join", "upload", "submit", "select_final", "download"],
            "budgets": {"money": 0, "gpu_seconds": 0, "wall_seconds": 10, "submissions": 3, "uploads": 3, "train_runs": 0},
            "max_daily_submissions": 3,
            "rules_acceptance": {"mode": "delegated", "allowed": True, "constraints": {
                "no_payment": True, "no_publication": True, "no_personal_disclosure": True}},
            "privacy": {"private_only": True, "no_publication": True, "no_personal_disclosure": True}}))
        self.store = MandateStore(self.root, mandate_path)
        self.adapter = CompetitionAdapter(self.config, self.store)
        self.join()
        first = self.adapter.submit(self.candidate())
        self.assertEqual(first["status"], "succeeded")
        self.assertEqual(self.adapter.submit(self.candidate()), first)
        self.assertEqual(self.store.inspect()["used"]["submissions"], 1)
        self.assertEqual(self.adapter.score(first)["score"], 1.0)

    def test_download_reserved_and_actual_bytes_read_back(self):
        public = self.root / "public-train.csv"
        public.write_text("id,target\na,1\n")
        self.config["data"] = {"train_path": str(public)}
        self.adapter = CompetitionAdapter(self.config, self.store)
        self.assertEqual(self.adapter.inspect()["data"]["train_path"], str(public))
        self.join()
        target = self.root / "download" / "train.csv"
        result = self.adapter.download({"files": [{"source_path": str(public), "destination": str(target)}]})
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(target.read_bytes(), public.read_bytes())
        self.assertEqual(result["files"][0]["sha256"], hashlib.sha256(target.read_bytes()).hexdigest())
        self.assertEqual(self.store.inspect()["receipts"][-1]["operation"], "download")

    def test_structural_notebook_archive_and_validation_receipt(self):
        candidate = self.candidate()
        path = self.root / "notebook.ipynb"
        path.write_text(json.dumps({"nbformat": 4, "cells": [{"cell_type": "code", "source": ["x = 1"]}]}))
        notebook = {**candidate, "path": str(path), "kind": "notebook", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.assertEqual(validate_candidate(notebook)["structural_validation"], "passed")
        with self.assertRaisesRegex(AdapterError, "receipt"):
            validate_candidate({**notebook, "validation_receipt": {"valid": True, "artifact_sha256": "stale"}})
        path = self.root / "agent.tar"
        with tarfile.open(path, "w") as archive:
            content = b"def agent(observation):\n    return 1\n"
            info = tarfile.TarInfo("main.py")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        agent = {**candidate, "path": str(path), "kind": "agent", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.assertEqual(validate_candidate(agent)["structural_validation"], "passed")
        with tarfile.open(path, "w") as archive:
            info = tarfile.TarInfo("../escape")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
        agent["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(AdapterError, "unsafe"):
            validate_candidate(agent)


if __name__ == "__main__":
    unittest.main()
