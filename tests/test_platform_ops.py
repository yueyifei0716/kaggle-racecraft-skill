"""Offline tests: all CLI and MCP traffic is mocked or a local fake executable."""

import contextlib
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import platform_ops as ops
import runtime


class TemporaryCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def approval(self, request, maximum=1, seconds=3600):
        path = self.root / "approval.json"
        ops.atomic_json(path, {"request": request,
            "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat(),
            "quota": {"max_operations": maximum}})
        return path

    def request(self):
        _, submission = self.csv_pair()
        return {"operation": "submit-file", "competition": "example", "artifact_sha256": ops.file_sha(submission),
                "file_name": "submission.csv", "message": "approved text"}

    def csv_pair(self):
        sample = self.write("sample.csv", "id,pred\n1,0\n2,0\n")
        submission = self.write("submission.csv", "id,pred\n1,0.2\n2,0.8\n")
        return sample, submission


class PackageTests(TemporaryCase):
    def test_credentials_never_read_secret_values(self):
        secret = self.write("auth/kaggle.json", "DO_NOT_READ_SECRET")
        secret.chmod(0o644)
        with patch.object(Path, "read_text", side_effect=AssertionError("credential read")):
            result = ops.credential_check(secret.parent, {"KAGGLE_KEY": "PRIVATE"})
        self.assertFalse(result["files"][0]["owner_only"])
        self.assertNotIn("PRIVATE", json.dumps(result))
        self.assertNotIn(str(secret.parent), json.dumps(result))

    def test_credential_permissions_repair_explicit_and_symlink_refused(self):
        secret = self.write("auth/kaggle.json", "never read value")
        secret.chmod(0o644)
        parser = ops.build_parser()
        dry = parser.parse_args(["credentials", "--config-dir", str(secret.parent), "--fix-perms"])
        self.assertEqual(ops.dispatch(dry)["permission_repair"], "dry_run")
        self.assertEqual(secret.stat().st_mode & 0o777, 0o644)
        with patch.object(Path, "read_text", side_effect=AssertionError("credential read")):
            execute = parser.parse_args(["credentials", "--config-dir", str(secret.parent), "--fix-perms", "--execute"])
            ops.dispatch(execute)
        self.assertEqual(secret.stat().st_mode & 0o777, 0o600)
        secret.unlink()
        target = self.write("target", "secret")
        target.chmod(0o644)
        secret.symlink_to(target)
        with self.assertRaises(ops.PlatformError):
            ops.dispatch(execute)
        self.assertEqual(target.stat().st_mode & 0o777, 0o644)

    def test_private_kernel_manifest_and_wrapper(self):
        source = self.write("input.py", 'from __future__ import annotations\nopen("result.txt", "w").write("ok")\n')
        out = self.root / "kernel"
        result = ops.prepare_kernel(source, out, "owner/private-kernel", "Private example")
        metadata = ops.read_json(out / "kernel-metadata.json")
        self.assertIs(metadata["is_private"], True)
        self.assertIs(metadata["enable_internet"], False)
        self.assertEqual(result["package_sha256"], ops.package_identity(out))
        subprocess.run([sys.executable, str(out / "main.py")], cwd=out, check=True, capture_output=True)
        self.assertEqual((out / "result.txt").read_text(), "ok")
        artifacts = ops.read_json(out / "run-artifacts.json")
        self.assertEqual(artifacts["log_file"], "runtime-events.jsonl")
        self.assertTrue(any(item["file"] == "result.txt" for item in artifacts["files"]))
        self.assertIn('"complete"', (out / "runtime-events.jsonl").read_text())

    def test_notebook_outputs_cleared_and_completion_explicit(self):
        notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [
            {"cell_type": "code", "execution_count": 3, "outputs": [{"secret": "old"}],
             "metadata": {}, "source": ["x=1"]}]}
        source = self.write("n.ipynb", json.dumps(notebook))
        out = self.root / "notebook"
        ops.prepare_kernel(source, out, "owner/private-kernel", "Example")
        result = ops.read_json(out / "notebook.ipynb")
        self.assertEqual(result["cells"][1]["outputs"], [])
        self.assertIn("platform_log_event", "".join(result["cells"][0]["source"]))
        self.assertIn("complete", "".join(result["cells"][-1]["source"]))

    def test_private_dataset_and_tamper_detection(self):
        self.write("source/weights.bin", "weights")
        out = self.root / "dataset"
        ops.prepare_dataset(self.root / "source", out, "owner/model-files", "Weights", "CC0-1.0")
        self.assertEqual(ops.read_json(out / "dataset-metadata.json")["id"], "owner/model-files")
        (out / "weights.bin").write_text("changed")
        with self.assertRaisesRegex(ops.PlatformError, "changed"):
            ops.package_identity(out)

    def test_upload_symlink_and_credentials_refused(self):
        self.write("data/plain.txt", "data")
        (self.root / "data/link").symlink_to(self.root / "data/plain.txt")
        with self.assertRaises(ops.PlatformError):
            ops.manifest(self.root / "data")
        (self.root / "data/link").unlink()
        self.write("data/kaggle.json", "do not upload")
        with self.assertRaises(ops.PlatformError):
            ops.manifest(self.root / "data")

    def test_prepare_dryrun_and_help_have_no_effect(self):
        parser = ops.build_parser()
        args = parser.parse_args(["prepare-kernel", "missing.py", "--out", str(self.root / "never"),
                                  "--id", "owner/example", "--title", "Test"])
        with patch.object(ops, "prepare_kernel", side_effect=AssertionError("mutation")), patch("subprocess.run", side_effect=AssertionError("network")):
            self.assertTrue(ops.dispatch(args)["dry_run"])
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exit:
                ops.main(["--help"])
            self.assertEqual(exit.exception.code, 0)
        self.assertFalse((self.root / "never").exists())


class RemoteTests(TemporaryCase):
    def test_cli_argv_no_shell_and_error_redaction(self):
        completed = subprocess.CompletedProcess([], 0, "ok", "")
        with patch("subprocess.run", return_value=completed) as run:
            self.assertEqual(ops.CLI().run(["competitions", "list", "-s", "$(secret)"]), "ok")
        self.assertFalse(run.call_args.kwargs["shell"])
        self.assertEqual(run.call_args.args[0][-1], "$(secret)")
        with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 1, "/private/path SECRET", "TOKEN")):
            with self.assertRaises(ops.PlatformError) as error:
                ops.CLI().run(["anything"])
        self.assertNotIn("SECRET", str(error.exception))
        self.assertNotIn("TOKEN", str(error.exception))

    def test_cli_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("kaggle", 1)):
            with self.assertRaisesRegex(ops.PlatformError, "unknown"):
                ops.CLI(timeout=1).run(["kernels", "push"])

    def test_wait_backoff_bounded(self):
        cli = Mock()
        cli.run.return_value = 'owner/example/7 has status "running"'
        elapsed, pauses = [0], []
        def sleep(seconds):
            pauses.append(seconds)
            elapsed[0] += seconds
        with self.assertRaisesRegex(ops.PlatformError, "timed out"):
            ops.wait_kernel(cli, "owner/example", 7, timeout=11, interval=2, max_interval=5,
                clock=lambda: elapsed[0], sleep=sleep)
        self.assertEqual(sum(pauses), 11)
        self.assertEqual(pauses[:2], [2, 3.4])
        self.assertLessEqual(max(pauses), 5)

    def test_terminal_error_and_version_mismatch_nonzero(self):
        cli = Mock()
        cli.run.return_value = 'owner/example/7 has status "error"'
        with self.assertRaisesRegex(ops.PlatformError, "unsuccessfully"):
            ops.wait_kernel(cli, "owner/example", 7)
        cli.run.return_value = 'owner/example/8 has status "complete"'
        with self.assertRaisesRegex(ops.PlatformError, "unrecognized"):
            ops.kernel_status(cli, "owner/example", 7)

    def test_old_cli_rejected_without_network(self):
        fake = Mock()
        fake.locate_file.return_value = self.write("old_api.py", "def kernels_output(kernel):\n    return None\n")
        fake.files = [Path("bin/kaggle")]
        with patch.object(ops.importlib.metadata, "distribution", return_value=fake):
            with self.assertRaisesRegex(ops.PlatformError, "exact-version"):
                ops.exact_version_capability(str(fake.locate_file.return_value))

    def test_exact_output_rechecks_hash_and_version(self):
        class FakeCLI:
            def run(self, arguments, timeout=None):
                if arguments[1] == "status":
                    return 'owner/example/7 has status "complete"'
                self.arguments = arguments
                Path(arguments[arguments.index("-p") + 1], "submission.csv").write_text("id,pred\n1,0.2\n2,0.8\n")
                return ""
        cli = FakeCLI()
        out = self.root / "download"
        remote = ops.fetch_output(cli, "owner/example", 7, out, capability=lambda: {"audited": True})
        self.assertEqual(cli.arguments[2], "owner/example/7")
        self.assertEqual(remote["version"], 7)
        sample, _ = self.csv_pair()
        validated = self.root / "validated.json"
        ops.validate_output(out, "owner/example", 7, "submission.csv", sample, ["id"], ["pred"], validated)
        self.assertEqual(len(ops.code_receipt(validated, "owner/example", 7, "submission.csv", out)), 64)
        with self.assertRaises(ops.PlatformError):
            ops.code_receipt(validated, "owner/example", 8, "submission.csv", out)
        (out / "submission.csv").write_text("changed")
        with self.assertRaisesRegex(ops.PlatformError, "bytes changed"):
            ops.code_receipt(validated, "owner/example", 7, "submission.csv", out)

    def test_output_pages_preserve_exact_version_and_unique_files(self):
        seen = []
        def run(arguments, timeout=None):
            if arguments[1] == "status":
                return 'owner/example/2 has status "complete"'
            token = arguments[arguments.index("--page-token") + 1]
            seen.append((arguments[2], token))
            folder = Path(arguments[arguments.index("-p") + 1])
            (folder / "shared.txt").write_text("same")
            (folder / ("first.csv" if not token else "second.csv")).write_text("data")
            return "Next page token: next-page" if not token else ""
        cli = Mock()
        cli.run.side_effect = run
        result = ops.fetch_output(cli, "owner/example", 2, self.root / "output", lambda: {})
        self.assertEqual(result["download_pages"], 2)
        self.assertEqual(len(result["files"]), 3)
        self.assertEqual(seen, [("owner/example/2", ""), ("owner/example/2", "next-page")])

    def test_output_repeated_pagination_is_rejected(self):
        def run(arguments, timeout=None):
            if arguments[1] == "status":
                return 'owner/example/2 has status "complete"'
            folder = Path(arguments[arguments.index("-p") + 1])
            (folder / "shared.txt").write_text("same")
            return "Next page token: repeat"
        cli = Mock()
        cli.run.side_effect = run
        with self.assertRaisesRegex(ops.PlatformError, "repeated"):
            ops.fetch_output(cli, "owner/example", 2, self.root / "never", lambda: {})
        self.assertFalse((self.root / "never").exists())

    def test_zip_slip_absolute_and_symlink_refused_before_writes(self):
        for index, name in enumerate(("../outside", "/absolute", "normal\\..\\outside")):
            archive = self.root / f"{index}.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("valid", "safe")
                handle.writestr(name, "bad")
            out = self.root / f"unzip{index}"
            with self.assertRaises(ops.PlatformError):
                ops.safe_extract_zip(archive, out)
            self.assertFalse(out.exists())

    def test_zip_extract_valid(self):
        archive = self.root / "valid.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("nested/result.csv", "ok")
        ops.safe_extract_zip(archive, self.root / "unzip")
        self.assertEqual((self.root / "unzip/nested/result.csv").read_text(), "ok")

    def test_fake_executable_integration_no_kaggle_service(self):
        executable = self.write("fake-kaggle", '#!' + sys.executable + '\nimport sys\nassert sys.argv[1:]==["kernels","status","owner/example/3"]\nprint(\'owner/example/3 has status "complete"\')\n')
        executable.chmod(0o700)
        self.assertEqual(ops.kernel_status(ops.CLI(str(executable)), "owner/example", 3)["status"], "complete")


class LedgerTests(TemporaryCase):
    def test_pending_before_network_and_duplicate_unknown_block(self):
        ledger = ops.Ledger(self.root / "ledger")
        request = self.request()
        approval = self.approval(request)
        def fail(arguments):
            receipt = ledger.receipts()[0]
            self.assertEqual(receipt["state"], "pending")
            self.assertEqual((ledger.folder / f'{receipt["attempt_id"]}.json').stat().st_mode & 0o777, 0o600)
            raise ops.PlatformError("network uncertain")
        cli = Mock()
        cli.run.side_effect = fail
        with self.assertRaises(ops.PlatformError):
            ops.execute_mutation(cli, ["competitions", "submit", "-f", self.root / "submission.csv"], request, ledger, approval)
        self.assertEqual(ledger.receipts()[0]["state"], "unknown")
        with self.assertRaisesRegex(ops.PlatformError, "duplicate"):
            ledger.reserve({**request, "message": "new message", "file_name": "renamed.csv"}, approval)
        self.assertEqual(cli.run.call_count, 1)

    def test_approval_exact_body_expiry_and_quota(self):
        request = self.request()
        approval = self.approval(request)
        with self.assertRaisesRegex(ops.PlatformError, "match"):
            ops.approval_for(approval, {**request, "message": "changed"}, [])
        expired = self.approval(request, seconds=-1)
        with self.assertRaisesRegex(ops.PlatformError, "expired"):
            ops.approval_for(expired, request, [])
        approval = self.approval(request)
        sha = ops.approval_for(approval, request, [])
        with self.assertRaisesRegex(ops.PlatformError, "quota"):
            ops.approval_for(approval, request, [{"approval_sha256": sha}])

    def test_success_ref_reconcile_json_and_blank_score_failure(self):
        request = self.request()
        ledger = ops.Ledger(self.root / "ledger")
        cli = Mock()
        cli.run.return_value = "Submission ref: 123\nSuccess"
        receipt = ops.execute_mutation(cli, ["competitions", "submit", "-f", self.root / "submission.csv"], request, ledger, self.approval(request))
        self.assertEqual(receipt["submission_ref"], "123")
        row = {"ref": 123, "status": "complete", "description": "approved text", "fileName": "submission.csv", "publicScore": ""}
        cli.run.return_value = json.dumps([row])
        with self.assertRaisesRegex(ops.PlatformError, "blank"):
            ops.reconcile(cli, ledger, receipt["attempt_id"])
        self.assertEqual(ledger.receipts()[0]["state"], "score_unavailable")
        cli.run.return_value = json.dumps([{**row, "publicScore": "0"}])
        self.assertEqual(ops.reconcile(cli, ledger, receipt["attempt_id"])["public_score"], 0)

    def test_csv_scores_and_remote_error(self):
        request = self.request()
        ledger = ops.Ledger(self.root / "ledger")
        receipt = ledger.reserve(request, self.approval(request))
        receipt = ledger.update(receipt, state="submitted", submission_ref="42")
        cli = Mock()
        cli.run.return_value = "ref,status,description,fileName,publicScore\n42,complete,approved text,submission.csv,0.7\n"
        self.assertEqual(ops.reconcile(cli, ledger, receipt["attempt_id"])["public_score"], .7)
        cli.run.return_value = "ref,status,description,fileName,publicScore\n42,error,approved text,submission.csv,\n"
        with self.assertRaisesRegex(ops.PlatformError, "failed"):
            ops.reconcile(cli, ledger, receipt["attempt_id"])

    def test_unknown_only_suggests_candidate_ref(self):
        request = self.request()
        ledger = ops.Ledger(self.root / "ledger")
        receipt = ledger.reserve(request, self.approval(request))
        cli = Mock()
        cli.run.return_value = json.dumps([{"ref": 42, "status": "complete", "description": "approved text", "fileName": "submission.csv", "publicScore": ".9"}])
        result = ops.reconcile(cli, ledger, receipt["attempt_id"])
        self.assertEqual(result["state"], "unknown")
        self.assertNotIn("public_score", result)
        self.assertEqual(result["candidate_ref"], "42")

    def test_submit_file_dryrun_checks_csv_no_receipt_or_network(self):
        sample, submission = self.csv_pair()
        args = ops.build_parser().parse_args(["submit-file", "--competition", "example", "--file", str(submission),
            "--sample", str(sample), "--id-column", "id", "--numeric-column", "pred", "--message", "exact text",
            "--ledger", str(self.root / "never")])
        with patch.object(ops.CLI, "run", side_effect=AssertionError("network")):
            result = ops.dispatch(args)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["request"]["artifact_sha256"], ops.digest(submission.read_bytes()))
        self.assertFalse((self.root / "never").exists())

    def test_private_dataset_create_command_never_public(self):
        self.write("source/a.bin", "data")
        out = self.root / "package"
        ops.prepare_dataset(self.root / "source", out, "owner/private-data", "Private", "CC0-1.0")
        args = ops.build_parser().parse_args(["dataset-create", str(out), "--execute", "--approval", str(self.root / "approval.json"), "--ledger", str(self.root / "ledger")])
        request = {"operation": "dataset-create", "id": "owner/private-data", "package_sha256": ops.package_identity(out), "private": True, "message": ""}
        self.approval(request)
        with patch.object(ops.CLI, "run", return_value="Your private Dataset is being created.") as run:
            ops.dispatch(args)
        self.assertNotIn("--public", run.call_args.args[0])

    def test_original_csv_mutation_during_reservation_cannot_change_uploaded_bytes(self):
        request = self.request()
        ledger = ops.Ledger(self.root / "ledger")
        original_reserve = ledger.reserve
        original = self.root / "submission.csv"
        captured = []
        def reserve(*args):
            receipt = original_reserve(*args)
            original.write_text("id,pred\n1,.99\n2,.01\n")
            return receipt
        cli = Mock()
        def send(arguments):
            sent = Path(arguments[arguments.index("-f") + 1])
            captured.append(ops.file_sha(sent))
            self.assertEqual(sent.name, "submission.csv")
            self.assertNotEqual(sent, original)
            return "Submission ref: 123"
        cli.run.side_effect = send
        with patch.object(ledger, "reserve", side_effect=reserve):
            result = ops.execute_mutation(cli, ["competitions", "submit", "-f", original], request, ledger, self.approval(request))
        self.assertEqual(result["state"], "submitted")
        self.assertEqual(captured, [request["artifact_sha256"]])
        self.assertNotEqual(ops.file_sha(original), captured[0])

    def test_package_mutation_during_reservation_cannot_change_upload(self):
        self.write("source/a.bin", "approved")
        folder = self.root / "package"
        ops.prepare_dataset(self.root / "source", folder, "owner/example-data", "Example", "CC0-1.0")
        request = {"operation": "dataset-create", "id": "owner/example-data", "package_sha256": ops.package_identity(folder), "message": "", "private": True}
        ledger = ops.Ledger(self.root / "ledger")
        original_reserve = ledger.reserve
        def reserve(*args):
            receipt = original_reserve(*args)
            (folder / "a.bin").write_text("unapproved")
            return receipt
        cli = Mock()
        def send(arguments):
            frozen = Path(arguments[arguments.index("-p") + 1])
            self.assertNotEqual(frozen, folder)
            self.assertEqual(ops.package_identity(frozen), request["package_sha256"])
            return "Your private Dataset is being created."
        cli.run.side_effect = send
        with patch.object(ledger, "reserve", side_effect=reserve):
            ops.execute_mutation(cli, ["datasets", "create", "-p", folder], request, ledger, self.approval(request))

    def test_output_validation_cannot_bind_bytes_absent_from_download(self):
        sample, artifact = self.csv_pair()
        remote = {"kernel": "owner/example", "version": 7, "files": [{"file": "submission.csv", "sha256": ops.file_sha(artifact)}]}
        ops.atomic_json(self.root / "output-receipt.json", remote)
        original = ops.check_csv
        def mutate_original(*args):
            artifact.write_text("id,pred\n1,.99\n2,.01\n")
            return original(*args)
        validated = self.root / "validated.json"
        with patch.object(ops, "check_csv", side_effect=mutate_original):
            result = ops.validate_output(self.root, "owner/example", 7, "submission.csv", sample, ["id"], ["pred"], validated)
        self.assertEqual(result["artifact_sha256"], remote["files"][0]["sha256"])
        with self.assertRaises(ops.PlatformError):
            ops.code_receipt(validated, "owner/example", 7, "submission.csv", self.root)


class MCPTests(TemporaryCase):
    def test_dynamic_tools_and_token_not_returned(self):
        calls = []
        def transport(payload, headers):
            calls.append((payload, dict(headers)))
            method = payload["method"]
            if method == "notifications/initialized":
                return {}, "session"
            result = {"protocolVersion": "2025-03-26"} if method == "initialize" else (
                {"tools": [{"name": "actual_search", "annotations": {"readOnlyHint": True}}]} if method == "tools/list" else {"content": [{"type": "text", "text": "result"}]})
            return {"jsonrpc": "2.0", "id": payload["id"], "result": result}, "session"
        with patch.dict(os.environ, {"TEST_TOKEN": "secret-value"}):
            client = ops.MCP("https://example.invalid/mcp", "TEST_TOKEN", transport=transport)
        client.initialize()
        result = client.call_readonly("actual_search", {"query": "competition"})
        self.assertNotIn("secret-value", json.dumps(result))
        self.assertEqual(calls[-1][0]["method"], "tools/call")
        self.assertEqual(calls[0][1]["Authorization"], "Bearer secret-value")

    def test_mcp_unannotated_or_unknown_tool_refused(self):
        client = ops.MCP("https://example.invalid/mcp", transport=Mock())
        with patch.object(client, "tools", return_value=[{"name": "write"}]), patch.object(client, "rpc") as rpc:
            with self.assertRaises(ops.PlatformError):
                client.call_readonly("write", {})
            rpc.assert_not_called()
        with self.assertRaises(ops.PlatformError):
            ops.MCP("https://user:password@example.invalid/mcp")

    def test_mcp_id_mismatch_or_error(self):
        client = ops.MCP("https://example.invalid/mcp", transport=lambda p,h: ({"jsonrpc": "2.0", "id": -1, "result": {}}, None))
        with self.assertRaises(ops.PlatformError):
            client.rpc("tools/list")

    def test_sse_filters_notifications_and_returns_matching_result(self):
        client = ops.MCP("https://example.invalid/mcp")
        stream = io.BytesIO(b': heartbeat\n\ndata: {"jsonrpc":"2.0","method":"notifications/progress"}\n\n'
            b'event: message\ndata: {"jsonrpc":"2.0","id":3,"result":{"tools":[]}}\n\n')
        self.assertEqual(client._sse_result(stream, 3)["result"], {"tools": []})
        with self.assertRaisesRegex(ops.PlatformError, "without matching"):
            client._sse_result(io.BytesIO(b"data: {}\n\n"), 3)

    def test_intel_dryrun_never_invokes_cli(self):
        args = ops.build_parser().parse_args(["intel", "search", "--query", "example"])
        with patch.object(ops.CLI, "run", side_effect=AssertionError("network")):
            self.assertTrue(ops.dispatch(args)["dry_run"])


class RuntimeTests(TemporaryCase):
    def test_startup_p95_hidden_fold_tta_crash_projection(self):
        result = runtime.project(10, [1, 2, 3, 4], 2, 100, folds=2, tta=3,
                                 budget_seconds=2000, crash_seconds=5, safety_factor=1)
        self.assertEqual(result["p95_batch_seconds"], 4)
        self.assertEqual(result["projected_seconds"], 1215)
        self.assertTrue(result["within_budget"])
        with self.assertRaises(ValueError):
            runtime.project(0, [float("nan")], 1, 2)

    def timing_csv(self):
        return self.write("timings.csv", "config,seconds,passes,fold,tta,decode_seconds,unit_index,rss_mb,gpu_mb\n"
            "small,2,2,0,0,0.5,1,100,200\nsmall,4,2,0,0,0.5,2,110,210\n"
            "small,6,2,1,0,0.5,1,100,200\nsmall,8,2,1,0,0.5,2,110,210\n")

    def test_csv_p95_decode_loop_order_and_memory_growth(self):
        path = self.timing_csv()
        unit = runtime.project_csv(path, 10, [0, 1], [0], safety_factor=1,
            startup_seconds=3, finalize_seconds=2, queue_seconds=5,
            crash_min_seconds=1, crash_max_seconds=10, rss_limit_mb=150, config="small")
        fold = runtime.project_csv(path, 10, [0, 1], [0], "fold-outer", safety_factor=1, config="small")
        self.assertEqual(unit["inference_seconds"], 60)
        self.assertEqual(unit["decode_seconds"], 5)
        self.assertEqual(fold["decode_seconds"], 10)
        self.assertEqual(unit["projected_runtime_interval_seconds"], [71, 80])
        self.assertEqual(unit["projected_wall_interval_seconds"], [76, 85])
        self.assertEqual(unit["rss"]["projected_peak_mb"], 190)
        self.assertFalse(unit["rss"]["within_budget"])

    def test_csv_missing_phase_and_decode_fail(self):
        with self.assertRaisesRegex(ValueError, "unmeasured"):
            runtime.project_csv(self.timing_csv(), 10, [5], [0])
        missing = self.write("missing.csv", "seconds,passes,fold,tta\n1,1,0,0\n")
        with self.assertRaisesRegex(ValueError, "decode"):
            runtime.project_csv(missing, 10, [0], [0])

    def test_benchmark_measures_driver_and_grid(self):
        driver = self.write("driver.py", "def decode(item, config):\n    return bytes(item, 'utf8')\ndef infer(decoded, fold, tta, config):\n    return sum(decoded)\n")
        out = self.root / "measurements"
        result = runtime.benchmark(driver, ["one", "two"], [{"name": "base", "folds": [0, 1], "tta": [0]}], out)
        self.assertEqual(result["measured_rows"], 4)
        rows = runtime.read_timings(out / "timings.csv")
        self.assertEqual(len(rows), 4)
        self.assertGreaterEqual(rows[0]["decode_seconds"], 0)
        self.assertTrue(ops.read_json(out / "overheads.json")[0]["startup_seconds"] >= 0)

    def test_benchmark_dryrun_does_not_import_driver(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            runtime.main(["benchmark", "--driver", "does-not-exist", "--items", "missing", "--grid", "missing", "--out", str(self.root / "never")])
        self.assertTrue(json.loads(output.getvalue())["dry_run"])
        self.assertFalse((self.root / "never").exists())


if __name__ == "__main__":
    unittest.main()
