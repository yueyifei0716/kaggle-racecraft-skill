import csv
import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


workspace = load("workspace")
paired = load("paired_check")
submission = load("submission_check")


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def table(self, name, rows):
        path = self.root / name
        with path.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
        return path

    def scores(self, delta):
        base = {str(i): (str(i % 3), float(i)) for i in range(30)}
        candidate = {key: (fold, score + delta) for key, (fold, score) in base.items()}
        return base, candidate

    def test_metric_direction_and_harm(self):
        base, candidate = self.scores(.2)
        self.assertEqual(paired.compare(base, candidate, "higher", .1, 100, 1, 1)["statistical_screen"], "PASS")
        self.assertEqual(paired.compare(base, candidate, "lower", .1, 100, 1, 1)["statistical_screen"], "BLOCK")

    def test_identical_is_not_improvement(self):
        base, candidate = self.scores(0)
        self.assertEqual(paired.compare(base, candidate, "higher", .01, 100, 1, 1)["statistical_screen"], "BLOCK")

    def test_missing_unit_or_fold_mismatch(self):
        base, candidate = self.scores(.2)
        candidate.pop("0")
        with self.assertRaises(ValueError):
            paired.compare(base, candidate, "higher", .1, 100, 1, 1)
        candidate["0"] = ("wrong", .2)
        with self.assertRaises(ValueError):
            paired.compare(base, candidate, "higher", .1, 100, 1, 1)

    def test_fold_regression_blocks_global_gain(self):
        base, candidate = self.scores(.2)
        candidate = {key: (fold, score - .3 if fold == "0" else score + .5) for key, (fold, score) in base.items()}
        result = paired.compare(base, candidate, "higher", .01, 100, 1, 1)
        self.assertIn("insufficient agreement across folds", result["failures"])

    def test_invalid_score_inputs(self):
        for rows in ([['unit','fold','score'], ['a','0','nan'], ['b','1','0']],
                     [['unit','fold','score'], ['a','0','1'], ['a','1','2']],
                     [['unit','fold','score'], ['a','0','1','ignored'], ['b','1','2']],
                     [['unit','fold','score'], ['a','0'], ['b','1','2']]):
            with self.assertRaises(ValueError):
                paired.read_scores(self.table("scores.csv", rows))

    def test_csv_reordered_units_hash_and_ranges(self):
        sample = self.table("sample.csv", [["id", "p"], ["a", 0], ["b", 0]])
        candidate = self.table("out.csv", [["id", "p"], ["a", .1], ["b", .9]])
        result = submission.check(sample, candidate, ["id"], ["p"], 0, 1)
        self.assertEqual(len(result["sha256"]), 64)
        for rows in ([["id", "p"], ["b", .9], ["a", .1]],
                     [["id", "p"], ["a", "nan"], ["b", .9]],
                     [["id", "p"], ["a", .1], ["b", 1.1]],
                     [["id", "p"], ["a", .1], ["a", .2]]):
            with self.assertRaises(ValueError):
                submission.check(sample, self.table("bad.csv", rows), ["id"], ["p"], 0, 1)

    def test_compound_identifiers(self):
        rows = [["person", "side", "p"], ["a", "left", .2], ["a", "right", .5]]
        path = self.table("multi.csv", rows)
        self.assertEqual(submission.check(path, path, ["person", "side"], ["p"])["rows"], 2)

    def test_initialization_preserves_work(self):
        workspace.initialize(self.root)
        before = (self.root / "competition.json").read_bytes()
        with self.assertRaises(ValueError):
            workspace.initialize(self.root)
        self.assertEqual((self.root / "competition.json").read_bytes(), before)

    def test_append_events_and_timezone(self):
        workspace.initialize(self.root)
        event = self.root / "event.json"
        event.write_text(json.dumps({"id": "run-a", "event": "planned"}))
        workspace.record(self.root, event)
        event.write_text(json.dumps({"id": "run-a", "event": "failed"}))
        workspace.record(self.root, event)
        path = self.root / "competition.json"
        config = json.loads(path.read_text())
        config.update(deadline="2026-11-02T08:00:00+08:00", reserve_hours=5)
        path.write_text(json.dumps(config))
        result = workspace.inspect(self.root, datetime(2026, 11, 1, tzinfo=timezone.utc))
        self.assertEqual(result["hours_remaining"], 24)
        self.assertEqual(result["hours_before_safety_window"], 19)
        self.assertEqual(result["event_count"], 2)
        self.assertEqual(result["latest_events"]["run-a"]["event"], "failed")

    def test_corrupted_log_does_not_silently_disappear(self):
        workspace.initialize(self.root)
        (self.root / "experiments.jsonl").write_text('{"id":')
        with self.assertRaises(ValueError):
            workspace.inspect(self.root, datetime.now(timezone.utc))

    def test_append_without_trailing_newline(self):
        workspace.initialize(self.root)
        log = self.root / "experiments.jsonl"
        log.write_text(json.dumps({"id": "a", "event": "planned"}))
        event = self.root / "new.json"
        event.write_text(json.dumps({"id": "a", "event": "finished"}))
        workspace.record(self.root, event)
        result = workspace.inspect(self.root, datetime.now(timezone.utc))
        self.assertEqual(result["event_count"], 2)
        self.assertEqual(result["latest_events"]["a"]["event"], "finished")

    def test_append_refuses_corrupt_existing_log(self):
        workspace.initialize(self.root)
        log = self.root / "experiments.jsonl"
        log.write_text('{"id":')
        event = self.root / "new.json"
        event.write_text(json.dumps({"id": "a", "event": "planned"}))
        with self.assertRaises(ValueError):
            workspace.record(self.root, event)
        self.assertEqual(log.read_text(), '{"id":')


if __name__ == "__main__":
    unittest.main()
