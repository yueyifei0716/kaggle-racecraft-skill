import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mandate.py"
spec = importlib.util.spec_from_file_location("mandate_test_module", SCRIPT)
mandate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mandate)


class MandateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "authorization.json"
        self.value = {"schema_version": 1, "user_confirmed": True, "status": "active",
                      "scope": {"competition": "example", "url": "https://www.kaggle.com/competitions/example"},
                      "expires_at": "2099-01-01T00:00:00Z", "allowed_operations": sorted(mandate.OPERATIONS),
                      "budgets": {"money": 5, "gpu_seconds": 60, "wall_seconds": 120,
                                  "submissions": 4, "uploads": 3, "train_runs": 2},
                      "max_daily_submissions": 1,
                      "rules_acceptance": {"mode": "pinned", "allowed": True, "document_sha256": "a" * 64},
                      "privacy": {"private_only": True, "no_publication": True, "no_personal_disclosure": True}}
        self.write()
        self.store = mandate.MandateStore(self.root, self.path)
        self.now = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)

    def write(self):
        self.path.write_text(json.dumps(self.value))

    def reserve(self, identity="one", operation="submit", costs=None, now=None, metadata=None):
        return self.store.reserve(operation, "example", identity,
                                  {"submissions": 1} if costs is None else costs,
                                  metadata, now=now or self.now)

    def test_explicit_authorization_and_scope(self):
        for key in ("user_confirmed", "privacy", "rules_acceptance", "expires_at"):
            value = copy.deepcopy(self.value)
            value.pop(key)
            with self.subTest(key=key), self.assertRaises(mandate.MandateError):
                mandate.validate(value)
        for confirmed in (False, 1, "true", None):
            self.value["user_confirmed"] = confirmed
            with self.subTest(confirmed=confirmed), self.assertRaises(mandate.MandateError):
                mandate.validate(self.value)

    def test_invalid_costs_and_budgets(self):
        for invalid in (-1, float("nan"), float("inf"), True, "1"):
            with self.subTest(invalid=invalid), self.assertRaises(mandate.MandateError):
                self.reserve(costs={"submissions": 1, "money": invalid})
        for invalid in (1.0, 0, 2):
            with self.subTest(invalid=invalid), self.assertRaises(mandate.MandateError):
                self.reserve(costs={"submissions": invalid})
        self.value["budgets"]["money"] = -1
        with self.assertRaises(mandate.MandateError):
            mandate.validate(self.value)
        with self.assertRaises(mandate.MandateError):
            self.reserve(costs={"submissions": 1, "unexpected": 1})

    def test_expiry_operation_and_competition(self):
        with self.assertRaisesRegex(mandate.MandateError, "expired"):
            self.reserve(now=datetime(2099, 1, 1, tzinfo=timezone.utc))
        with self.assertRaisesRegex(mandate.MandateError, "scope"):
            self.store.reserve("read", "wrong", "one", {})
        with self.assertRaisesRegex(mandate.MandateError, "unsupported"):
            self.reserve(operation="publish", costs={})
        self.value["allowed_operations"] = ["read"]
        self.write()
        with self.assertRaisesRegex(mandate.MandateError, "not authorized"):
            self.reserve()

    def test_rules_version_and_acceptance(self):
        with self.assertRaisesRegex(mandate.MandateError, "rules"):
            self.reserve(operation="join", costs={})
        with self.assertRaisesRegex(mandate.MandateError, "changed"):
            self.reserve(metadata={"rules_document_sha256": "b" * 64})
        self.reserve(operation="join", costs={}, metadata={"rules_document_sha256": "a" * 64})
        root = self.root / "other"
        self.value["rules_acceptance"]["allowed"] = False
        self.write()
        with self.assertRaisesRegex(mandate.MandateError, "not authorized"):
            mandate.MandateStore(root, self.path).reserve("join", "example", "one", {},
                {"rules_document_sha256": "a" * 64})

    def test_canonical_identity_and_descriptions(self):
        first = self.reserve(identity={"sha": "artifact", "version": 1}, metadata={"description": "first"})
        again = self.reserve(identity={"version": 1, "sha": "artifact"}, metadata={"description": "new"})
        self.assertTrue(again["duplicate"])
        self.assertEqual(first["id"], again["id"])
        self.assertEqual(len(self.store.inspect()["receipts"]), 1)
        with self.assertRaisesRegex(mandate.MandateError, "different costs"):
            self.reserve(identity={"version": 1, "sha": "artifact"}, costs={"submissions": 1, "money": 1})

    def test_delegated_rules_require_constraints_and_documented_review(self):
        self.value["rules_acceptance"] = {"mode": "delegated", "allowed": True,
            "constraints": {"no_payment": True, "no_publication": True, "no_personal_disclosure": True}}
        self.write()
        metadata = {"rules_document_sha256": "c" * 64}
        for review in (None, {"complies": False, "evidence": "requires payment"},
                       {"complies": True, "evidence": ""},
                       {"complies": True, "evidence": "terms", "payment_required": True},
                       {"complies": True, "evidence": "terms", "no_publication": False}):
            metadata["rules_review"] = review
            with self.subTest(review=review), self.assertRaises(mandate.MandateError):
                self.reserve(operation="join", costs={}, metadata=metadata)
        metadata["rules_review"] = {"complies": True, "evidence": {"clauses": ["no fees", "private entry"]}}
        receipt = self.reserve(operation="join", costs={}, metadata=metadata)
        self.assertEqual(receipt["metadata"]["rules_document_sha256"], "c" * 64)
        self.value["rules_acceptance"]["constraints"]["no_payment"] = False
        with self.assertRaisesRegex(mandate.MandateError, "constraints"):
            mandate.validate(self.value)

    def test_completion_cannot_refund_operation_counts(self):
        receipt = self.reserve()
        self.store.complete(receipt["id"], {"accepted": True}, {"submissions": 0})
        self.assertEqual(self.store.inspect()["used"]["submissions"], 1)

    def test_completion_is_idempotent_and_releases_only_measured_cost(self):
        receipt = self.reserve(costs={"submissions": 1, "money": 3})
        completed = self.store.complete(receipt["id"], {"remote_id": 42}, {"money": 2})
        self.assertEqual(completed, self.store.complete(receipt["id"], {"remote_id": 42}, {"money": 2}))
        self.assertEqual(self.store.inspect()["used"]["money"], 2)
        self.assertEqual(self.store.inspect()["used"]["submissions"], 1)
        self.assertEqual(self.reserve(costs={"submissions": 1, "money": 3})["result"], {"remote_id": 42})
        with self.assertRaisesRegex(mandate.MandateError, "conflicts"):
            self.store.complete(receipt["id"], {"remote_id": 43})

    def test_unknown_and_pending_survive_restart_without_retry(self):
        receipt = self.reserve()
        self.store.fail(receipt["id"], "connection lost")
        restarted = mandate.MandateStore(self.root, self.path)
        again = restarted.reserve("submit", "example", "one", {"submissions": 1}, now=self.now)
        self.assertTrue(again["duplicate"])
        self.assertEqual(again["status"], "unknown")
        self.assertEqual(restarted.inspect()["used"]["submissions"], 1)
        self.assertEqual(restarted.complete(receipt["id"], {"remote_id": 1})["status"], "complete")
        pending = self.reserve(operation="train", identity="train", costs={"train_runs": 1, "gpu_seconds": 50})
        self.assertEqual(mandate.MandateStore(self.root, self.path).inspect()["used"]["gpu_seconds"], 50)
        self.assertEqual(pending["status"], "pending")

    def test_failed_still_occupies_budget(self):
        receipt = self.reserve()
        self.store.fail(receipt["id"], "rejected", unknown=False)
        with self.assertRaisesRegex(mandate.MandateError, "daily"):
            self.reserve(identity="two")
        with self.assertRaises(mandate.MandateError):
            self.store.complete(receipt["id"], {})

    def test_unknown_can_be_closed_after_definitive_not_found(self):
        receipt = self.reserve()
        self.store.fail(receipt["id"], "timeout")
        closed = self.store.fail(receipt["id"], "platform confirms operation not found", unknown=False)
        self.assertEqual(closed["status"], "failed")
        self.assertEqual(self.store.inspect()["used"]["submissions"], 1)
        self.assertTrue(self.reserve()["duplicate"])

    def test_daily_limit_is_utc_and_total_does_not_reset(self):
        self.reserve()
        with self.assertRaisesRegex(mandate.MandateError, "daily"):
            self.reserve(identity="two")
        self.reserve(identity="two", now=datetime(2026, 11, 2, tzinfo=timezone.utc))
        self.assertEqual(self.store.inspect()["used"]["submissions"], 2)
        with self.assertRaisesRegex(mandate.MandateError, "daily"):
            self.reserve(identity="three", now="2026-11-03T01:00:00+02:00")

    def test_overspend_blocks_and_is_durable(self):
        receipt = self.reserve(costs={"submissions": 1, "money": 1})
        result = self.store.complete(receipt["id"], {"ok": True}, {"money": 2})
        self.assertEqual(result["status"], "overspend")
        self.assertEqual(self.store.inspect()["used"]["money"], 2)
        with self.assertRaisesRegex(mandate.MandateError, "overspend"):
            self.reserve(operation="read", costs={}, identity="poll")
        self.assertEqual(self.store.complete(receipt["id"], {"ok": True}, {"money": 2}), result)

    def test_privacy_and_read_polling(self):
        for metadata in ({"private": False}, {"publication": True}, {"personal_disclosure": True}):
            with self.subTest(metadata=metadata), self.assertRaises(mandate.MandateError):
                self.reserve(metadata=metadata)
        for poll in ("00:00", "00:01"):
            self.reserve(identity={"poll": poll}, operation="read", costs={})
        self.assertEqual(self.store.inspect()["used"], dict.fromkeys(mandate.BUDGETS, 0))
        with self.assertRaisesRegex(mandate.MandateError, "zero"):
            self.reserve(operation="read", costs={"money": 1})

    def test_revoke_pause_and_authorization_edits(self):
        self.reserve()
        self.value["budgets"]["submissions"] = 5
        self.write()
        with self.assertRaisesRegex(mandate.MandateError, "mandate_changed"):
            self.store.inspect()
        self.value["budgets"]["submissions"] = 4
        self.write()
        self.store.pause("user paused")
        with self.assertRaisesRegex(mandate.MandateError, "paused"):
            self.reserve()
        self.store.revoke()
        with self.assertRaisesRegex(mandate.MandateError, "revoked"):
            self.reserve()

    def test_concurrent_processes_share_reservation_and_budget(self):
        command = [sys.executable, str(SCRIPT), "--root", str(self.root), "--mandate", str(self.path),
                   "reserve", "--operation", "submit", "--competition", "example",
                   "--identity", '"same"', "--costs", '{"submissions":1}']
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: subprocess.run(command, capture_output=True, text=True), range(8)))
        self.assertEqual([result.returncode for result in results], [0] * 8)
        receipts = [json.loads(result.stdout) for result in results]
        self.assertEqual(sum(not receipt["duplicate"] for receipt in receipts), 1)
        self.assertEqual(self.store.inspect()["used"]["submissions"], 1)
        events = [json.loads(line) for line in self.store.audit_file.read_text().splitlines()]
        self.assertEqual([event["event"] for event in events], ["reserve"])

    def test_concurrent_distinct_reservations_cannot_exceed_budget(self):
        def run(index):
            try:
                return self.reserve(identity=f"submission-{index}")
            except mandate.MandateError:
                return None
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(run, range(12)))
        self.assertEqual(sum(result is not None for result in results), 1)

    def test_invalid_json_and_api_inputs_fail_closed(self):
        for identity in (None, [], " ", {}, float("nan")):
            with self.subTest(identity=identity), self.assertRaises(mandate.MandateError):
                self.reserve(identity=identity)
        with self.assertRaises(mandate.MandateError):
            self.reserve(metadata={"nan": float("nan")})
        for identifier in (None, [], "", "not-a-receipt"):
            with self.subTest(identifier=identifier), self.assertRaises(mandate.MandateError):
                self.store.complete(identifier, {})
            with self.subTest(identifier=identifier), self.assertRaises(mandate.MandateError):
                self.store.fail(identifier, "unknown")
        self.path.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaisesRegex(mandate.MandateError, "duplicate JSON"):
            self.store.inspect()


if __name__ == "__main__":
    unittest.main()
