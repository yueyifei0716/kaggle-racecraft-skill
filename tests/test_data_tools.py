import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


data = load("data_tools", "scripts/data_tools.py")
inference = load("inference_support", "assets/inference_support.py")


class DataTest(unittest.TestCase):
    def rows(self):
        return [{"id": str(i), "group": str(i // 3), "label": str(i % 2), "time": str(i // 2)} for i in range(36)]

    def test_group_split_stable_and_disjoint(self):
        rows = self.rows()
        folds, _ = data.assign(rows, "id", "stratified-group", 3, group="group", target="label")
        second, _ = data.assign(rows, "id", "stratified-group", 3, group="group", target="label")
        self.assertEqual(folds, second)
        for group in set(row["group"] for row in rows):
            self.assertEqual(len({fold for fold, row in zip(folds, rows) if row["group"] == group}), 1)
        self.assertEqual(set(folds), {0, 1, 2})

    def test_temporal_training_precedes_validation_and_gap(self):
        rows = self.rows()
        folds, result = data.assign(rows, "id", "time", 3, time_column="time", gap=1)
        by_id = {row["id"]: int(row["time"]) for row in rows}
        self.assertIn(-1, folds)
        for plan in result["plan"]:
            self.assertLess(max(by_id[value] for value in plan["train_ids"]) + 1,
                            min(by_id[value] for value in plan["validation_ids"]))

    def test_leave_one_group_out(self):
        rows = self.rows()
        folds, report = data.assign(rows, "id", "leave-group-out", 5, group="group")
        groups = {row["group"] for row in rows}
        self.assertEqual(len(set(folds)), len(groups))
        for fold in set(folds):
            self.assertEqual(len({row["group"] for index, row in enumerate(rows) if folds[index] == fold}), 1)

    def test_holdout_groups_and_duplicate_rejection(self):
        dev, held = data.seal(self.rows(), "id", "group", .2, 17)
        self.assertFalse({row["group"] for row in dev} & {row["group"] for row in held})
        with self.assertRaises(ValueError):
            data.assign(self.rows() + [self.rows()[0]], "id", "random", 3)

    def test_time_values_nonfinite_and_equivalent_offsets(self):
        with self.assertRaises(ValueError):
            data.assign([dict(row, time="nan") for row in self.rows()], "id", "time", 2, time_column="time")
        rows = [{"id": "a", "time": "2026-01-01T00:00:00Z"},
                {"id": "b", "time": "2026-01-01T08:00:00+08:00"},
                {"id": "c", "time": "2026-01-02T00:00:00Z"},
                {"id": "d", "time": "2026-01-03T00:00:00Z"}]
        folds, _ = data.assign(rows, "id", "time", 2, time_column="time")
        self.assertEqual(folds[0], folds[1])

    def test_mixed_time_formats_rejected_before_future_leakage(self):
        rows = [{"id": "jan01", "time": "2026-01-01T00:00:00Z"},
                {"id": "jan03", "time": "20260103"},
                {"id": "jan04", "time": "2026-01-04T00:00:00Z"}]
        with self.assertRaisesRegex(ValueError, "mixes"):
            data.assign(rows, "id", "time", 2, time_column="time")

    def test_plan_csv_alias_rejected_without_partial_output(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data.csv"
            source.write_text("id,time\na,1\nb,2\nc,3\n")
            output = root / "same.csv"
            result = subprocess.run([sys.executable, str(ROOT / "scripts/data_tools.py"), "folds", str(source),
                                     str(output), "--strategy", "time", "--id", "id", "--time", "time",
                                     "--splits", "2", "--plan", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(output.exists())

    def test_labels_probabilities_and_variable_rows(self):
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        import submission_check
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "sample.csv"
            output = root / "output.csv"
            sample.write_text("id,label\na,cat\n")
            output.write_text("id,label\nb,dog\nc,cat\n")
            result = submission_check.check(sample, output, ["id"], [], mode="variable",
                                            label_columns=["label"], allowed_labels=["cat", "dog"])
            self.assertEqual(result["rows"], 2)
            output.write_text("id,label\nb,unknown\n")
            with self.assertRaises(ValueError):
                submission_check.check(sample, output, ["id"], [], mode="variable",
                                       label_columns=["label"], allowed_labels=["cat", "dog"])

    def test_atomic_output_preserves_previous_on_invalid_row(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.csv"
            output.write_text("original")
            with self.assertRaises(ValueError):
                inference.atomic_csv(output, ["id", "p"], [{"id": "a", "p": float("nan")}], ["p"])
            self.assertEqual(output.read_text(), "original")
            result = inference.atomic_csv(output, ["id", "p"], [{"id": "a", "p": .8}], ["p"])
            self.assertEqual(result["sha256"], hashlib.sha256(output.read_bytes()).hexdigest())

    def test_artifact_path_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model").write_bytes(b"weights")
            good = {"path": "model", "sha256": hashlib.sha256(b"weights").hexdigest()}
            self.assertEqual(inference.verify_artifacts(root, [good]), ["model"])
            for item in ({"path": "../model", "sha256": good["sha256"]}, dict(good, sha256="bad")):
                with self.assertRaises(ValueError):
                    inference.verify_artifacts(root, [item])

    def test_budget_and_explicit_fallback(self):
        clock = [0]
        budget = inference.RunBudget(10, 2, lambda: clock[0])
        self.assertTrue(budget.optional_pass_fits(7))
        clock[0] = 9
        with self.assertRaises(TimeoutError):
            budget.require()
        with tempfile.TemporaryDirectory() as directory:
            def broken(item):
                raise FileNotFoundError("model missing")
            with self.assertRaises(FileNotFoundError):
                inference.predict_items([1], broken, Path(directory) / "report.jsonl")


if __name__ == "__main__":
    unittest.main()
