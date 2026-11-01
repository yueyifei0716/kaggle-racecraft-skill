import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("agent_process", ROOT / "scripts/agent_process.py")
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


class AgentProcessTest(unittest.TestCase):
    def test_real_subprocess_argv_json_contract_and_no_secret_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "fake-agent"
            executable.write_text("#!/usr/bin/env python3\nimport sys,json\nfrom pathlib import Path\na=sys.argv\nassert '-m' in a and '--ephemeral' in a and '--dangerously-bypass-approvals-and-sandbox' not in a\nsys.stdin.read()\nprint('PRIVATE-LOG-NOT-RETURNED')\nPath(a[a.index('-o')+1]).write_text(json.dumps({'kind':'wait','reason':'test','seconds':1}))\n")
            executable.chmod(0o700)
            result = agent.run_agent("worker", {"task_kind": "plan"}, cli=executable, model="gpt-6.1-sol",
                                     workspace=root, skill=ROOT / "SKILL.md", timeout=5)
            self.assertEqual(result["kind"], "wait")
            self.assertNotIn("PRIVATE", json.dumps(result))

    def test_mutation_requires_operation_identity_before_invoking_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                agent.run_agent("platform", {"protocol": agent.PLATFORM_PROTOCOL, "action": "submit", "competition": "test"},
                                cli="must-not-run", model="gpt-6.1-sol", workspace=directory, skill=ROOT / "SKILL.md")

    def test_strict_response_and_old_model_rejected(self):
        for text in ('{"kind":"wait","kind":"stop"}', '{"cost":NaN}', '[]'):
            with self.assertRaises(ValueError):
                agent.strict_json(text)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                agent.run_agent("worker", {}, cli="must-not-run", model="gpt-5.6-sol", workspace=directory, skill=ROOT / "SKILL.md")


if __name__ == "__main__":
    unittest.main()
