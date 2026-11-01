"""Cross-module acceptance using a synthetic competition and no external services."""

import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.work = self.root / "competition"
        self.env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")

    def json_file(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def csv_file(self, name, header, rows):
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def cli(self, script, *args, expected=0):
        result = subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)],
                                env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, expected, result.stderr or result.stdout)
        return json.loads(result.stdout) if expected == 0 else result.stderr

    def state(self, action, payload=None, identifier=None, artifact=None, expected=0):
        args = [action, self.work]
        if payload is not None:
            args += ["--input", self.json_file("request.json", payload)]
        if identifier is not None:
            args += ["--id", identifier]
        if artifact is not None:
            args += ["--artifact", artifact]
        return self.cli("workspace.py", *args, expected=expected)

    def test_preregister_evaluate_promote_score_select_and_invalidate(self):
        self.state("init")
        stages = ["data", "features", "training", "prediction", "scoring"]
        self.state("config", {"competition": "synthetic-local", "metric": {"name": "mae", "direction": "lower"},
                              "public_units": 2,
                              "gates": {key: {"decision": "PASS", "reason": "synthetic fixture reviewed"}
                                        for key in ("strategy", "test_distribution", "validation", "midpoint", "finals")}})
        self.state("validator", {"authority": "full", "covered_stages": stages})
        self.state("register", {"id": "candidate-a", "family": "deterministic-fixture", "structural": True,
                                "swept": False, "stages": stages,
                                "hypothesis": "fixed prediction offset decreases absolute error"})
        base_rows = self.csv_file("base_rows.csv", ["row_id", "unit", "fold", "group", "y_true", "y_pred"],
                                  [[str(i), str(i), str(i % 3), str(i % 2), i, i + 2] for i in range(12)])
        candidate_rows = self.csv_file("candidate_rows.csv", ["row_id", "unit", "fold", "group", "y_true", "y_pred"],
                                       [[str(i), str(i), str(i % 3), str(i % 2), i, i + 1] for i in range(12)])
        base_units = self.csv_file("base_units.csv", ["unit", "score", "weight", "fold", "group"],
                                   [[str(i), 2, 1, str(i % 3), str(i % 2)] for i in range(12)])
        candidate_units = self.csv_file("candidate_units.csv", ["unit", "score", "weight", "fold", "group"],
                                        [[str(i), 1, 1, str(i % 3), str(i % 2)] for i in range(12)])
        sample = self.csv_file("sample.csv", ["id", "prediction"], [[str(i), 0] for i in range(12)])
        submission = self.csv_file("submission.csv", ["id", "prediction"], [[str(i), i + 1] for i in range(12)])
        self.state("update", identifier="candidate-a", artifact=submission)
        gate = self.cli("analysis.py", "gate", "--ledger", self.work / "ledger.json",
                        "--baseline", base_units, "--candidate", candidate_units,
                        "--baseline-rows", base_rows, "--candidate-rows", candidate_rows,
                        "--artifact", submission, "--min-gain", ".5", "--resamples", "100")
        self.assertEqual(gate["decision"], "PASS")
        sha = hashlib.sha256(submission.read_bytes()).hexdigest()
        self.state("score", {"artifact_sha256": sha, "gate": gate, "oof": 1}, "candidate-a")
        self.state("update", {"status": "promoted", "selection_units": {str(i): 1 for i in range(12)}}, "candidate-a")
        checked = self.cli("submission_check.py", sample, submission, "--id-column", "id", "--numeric-column", "prediction")
        self.assertEqual(checked["sha256"], sha)
        plan = self.cli("platform_ops.py", "submit-file", "--competition", "synthetic-local",
                        "--message", "candidate-a offset", "--file", submission, "--sample", sample,
                        "--id-column", "id", "--numeric-column", "prediction")
        self.assertTrue(plan["dry_run"])
        self.assertEqual(plan["request"]["artifact_sha256"], sha)
        # A simulated returned score enters the same real state API, never a live submission.
        self.state("score", {"artifact_sha256": sha, "public": .9}, "candidate-a")
        trust = self.cli("analysis.py", "trust", "--ledger", self.work / "ledger.json")
        self.assertEqual(trust["regime"], "INSUFFICIENT")
        self.state("trust", trust)
        selected = self.cli("analysis.py", "select", "--ledger", self.work / "ledger.json", "--count", "1")
        self.assertEqual([item["id"] for item in selected["recommendations"]], ["candidate-a"])
        self.state("final", {"candidate_ids": ["candidate-a"], "reason": "current local evidence only",
                             "holdout_decision": {"decision": "not_used", "reason": "synthetic acceptance fixture"}})
        self.assertTrue(self.state("status")["final_current"])
        submission.write_text(submission.read_text().replace("0,1", "0,1.5"))
        self.state("update", identifier="candidate-a", artifact=submission)
        status = self.state("status")
        self.assertFalse(status["final_current"])
        candidate = status["ledger"]["candidates"][0]
        self.assertIsNone(candidate["gate"])
        self.assertIsNone(candidate["public"])


if __name__ == "__main__":
    unittest.main()
