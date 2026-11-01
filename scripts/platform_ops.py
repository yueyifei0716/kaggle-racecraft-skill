"""Explicit, private-by-default Kaggle CLI operations with durable attempt receipts."""

import argparse
import ast
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile

from submission_check import check as check_csv


class PlatformError(Exception):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".receipt-", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def safe_slug(value, paired=False):
    pattern = r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+" if paired else r"[A-Za-z0-9_-]+"
    if not re.fullmatch(pattern, value):
        raise PlatformError("invalid platform identifier")
    return value


def exact_kernel(kernel, version):
    safe_slug(kernel, paired=True)
    if not isinstance(version, int) or version < 1:
        raise PlatformError("a positive exact kernel version is required")
    return f"{kernel}/{version}"


def credential_check(config_dir=None, environ=None, fix_perms=False):
    """Inspect existence and modes only, never read the credential file."""
    environ = os.environ if environ is None else environ
    folder = Path(config_dir or environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    entries = []
    for name in ("kaggle.json", "access_token"):
        path = folder / name
        try:
            info = path.lstat()
            if fix_perms:
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                    raise PlatformError("permission repair requires a regular credential file owned by this user")
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                try:
                    opened = os.fstat(descriptor)
                    if (not stat.S_ISREG(opened.st_mode) or opened.st_uid != os.getuid()
                            or (opened.st_ino, opened.st_dev) != (info.st_ino, info.st_dev)):
                        raise PlatformError("credential changed during permission inspection")
                    os.fchmod(descriptor, 0o600)
                    info = os.fstat(descriptor)
                finally:
                    os.close(descriptor)
            entries.append({"kind": name, "exists": True,
                "regular_file": stat.S_ISREG(info.st_mode),
                "owner_only": stat.S_IMODE(info.st_mode) & 0o077 == 0,
                "owned_by_user": info.st_uid == os.getuid()})
        except FileNotFoundError:
            entries.append({"kind": name, "exists": False})
    return {"files": entries, "environment_presence": {
        key: bool(environ.get(key)) for key in
        ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY")},
        "authenticated": "not tested"}


def manifest(folder, upload=True):
    root = Path(folder)
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PlatformError("symlinks are not permitted in upload packages")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if upload and relative == "package-manifest.json":
                continue
            if upload and any(part.startswith(".") for part in path.relative_to(root).parts):
                raise PlatformError("hidden files are not permitted in upload packages")
            if upload and path.name.lower() in ("kaggle.json", "access_token", ".env"):
                raise PlatformError("credential-like file refused")
            result.append({"file": relative, "size": path.stat().st_size,
                           "sha256": file_sha(path)})
    if not result:
        raise PlatformError("empty package")
    return result


def package_identity(folder):
    actual = manifest(folder)
    expected = read_json(Path(folder) / "package-manifest.json")
    if expected.get("files") != actual or expected.get("sha256") != digest(canonical(actual)):
        raise PlatformError("package changed since preparation")
    return expected["sha256"]


def write_manifest(folder):
    files = manifest(folder)
    value = {"files": files, "sha256": digest(canonical(files))}
    atomic_json(Path(folder) / "package-manifest.json", value, exclusive=True)
    return value


LOGGER = '''import atexit as _atexit, json as _json, time as _time, hashlib as _hashlib
from pathlib import Path as _Path
_platform_start = _time.monotonic()
def platform_log_event(stage, **metrics):
    with open("runtime-events.jsonl", "a", encoding="utf-8") as _log:
        _log.write(_json.dumps({"stage": stage, "elapsed_seconds": _time.monotonic()-_platform_start, "metrics": metrics})+"\\n")
platform_log_event("startup")
def _platform_finalize():
    platform_log_event("process_exit", outcome="see completion event")
    _files = []
    for _file in sorted(_Path(".").rglob("*")):
        if _file.is_file() and not _file.is_symlink() and _file.name != "run-artifacts.json":
            _hash = _hashlib.sha256()
            with _file.open("rb") as _input:
                for _chunk in iter(lambda: _input.read(1024*1024), b""):
                    _hash.update(_chunk)
            _files.append({"file": str(_file), "sha256": _hash.hexdigest(), "size": _file.stat().st_size})
    _Path("run-artifacts.json").write_text(_json.dumps({"log_file": "runtime-events.jsonl", "files": _files}, indent=2))
_atexit.register(_platform_finalize)
'''


def prepare_kernel(source, out, kernel, title, competition=None, gpu=False,
                   internet=False, datasets=()):
    safe_slug(kernel, paired=True)
    if competition:
        safe_slug(competition)
    for dataset in datasets:
        safe_slug(dataset, paired=True)
    source, out = Path(source), Path(out)
    raw = source.read_text(encoding="utf-8")
    if source.suffix not in (".py", ".ipynb"):
        raise PlatformError("kernel source must be Python or an ipynb notebook")
    if source.suffix == ".ipynb":
        notebook = json.loads(raw)
        if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
            raise PlatformError("notebook format 4 required")
        def cell(code):
            return {"cell_type": "code", "metadata": {}, "execution_count": None,
                    "outputs": [], "source": code.splitlines(keepends=True)}
        for item in notebook["cells"]:
            if item.get("cell_type") == "code":
                item["outputs"] = []
                item["execution_count"] = None
        notebook["cells"].insert(0, cell(LOGGER))
        notebook["cells"].append(cell('platform_log_event("complete")\n'))
        code_name, code = "notebook.ipynb", json.dumps(notebook, indent=2)
        kind = "notebook"
    else:
        compile(raw, "source.py", "exec")
        code_name = "main.py"
        code = LOGGER + '\ntry:\n    exec(compile(' + repr(raw) + ', "source.py", "exec"), globals())\nexcept BaseException:\n    platform_log_event("failed")\n    raise\nelse:\n    platform_log_event("complete")\n'
        kind = "script"
    out.mkdir(parents=True, exist_ok=False)
    (out / code_name).write_text(code, encoding="utf-8")
    metadata = {"id": kernel, "title": title, "code_file": code_name,
        "language": "python", "kernel_type": kind, "is_private": True,
        "enable_gpu": bool(gpu), "enable_internet": bool(internet),
        "dataset_sources": list(datasets), "competition_sources": [competition] if competition else [],
        "kernel_sources": []}
    atomic_json(out / "kernel-metadata.json", metadata, exclusive=True)
    return {"prepared": "kernel", "package_sha256": write_manifest(out)["sha256"],
            "private": True, "internet": bool(internet)}


def prepare_dataset(source, out, dataset, title, license_name="CC0-1.0"):
    safe_slug(dataset, paired=True)
    source, out = Path(source), Path(out)
    if not source.is_dir() or out.resolve().is_relative_to(source.resolve()):
        raise PlatformError("dataset source must be a directory separate from destination")
    manifest(source)
    if (source / "dataset-metadata.json").exists() or (source / "package-manifest.json").exists():
        raise PlatformError("source contains reserved metadata names")
    shutil.copytree(source, out)
    atomic_json(out / "dataset-metadata.json", {"id": dataset, "title": title,
        "licenses": [{"name": license_name}]}, exclusive=True)
    return {"prepared": "dataset", "package_sha256": write_manifest(out)["sha256"],
            "private": True, "note": "creation command omits --public"}


class CLI:
    def __init__(self, executable="kaggle", timeout=60):
        if not math.isfinite(timeout) or timeout <= 0:
            raise PlatformError("CLI timeout must be positive and finite")
        self.executable, self.timeout = executable, timeout

    def run(self, arguments, timeout=None):
        try:
            result = subprocess.run([self.executable, *map(str, arguments)],
                shell=False, capture_output=True, text=True,
                timeout=min(self.timeout, timeout) if timeout is not None else self.timeout,
                check=False)
        except subprocess.TimeoutExpired:
            raise PlatformError("CLI timed out; remote result may be unknown") from None
        except OSError:
            raise PlatformError("CLI could not start") from None
        if result.returncode:
            raise PlatformError("CLI failed; raw output withheld to protect credentials and local paths")
        return result.stdout


def exact_version_capability(executable="kaggle"):
    """Fail closed on installed CLIs that may silently select the latest version."""
    try:
        distribution = importlib.metadata.distribution("kaggle")
        resolved = Path(shutil.which(executable) or executable).resolve()
        entrypoints = [Path(distribution.locate_file(item)).resolve() for item in distribution.files or []
                       if Path(str(item)).name in ("kaggle", "kaggle.exe")]
        if resolved not in entrypoints:
            raise PlatformError("selected executable does not match the inspected official CLI distribution")
        path = distribution.locate_file("kaggle/api/kaggle_api_extended.py")
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {node.name: ast.get_source_segment(source, node)
            for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        for name in ("kernels_output", "kernels_status"):
            body = functions.get(name, "")
            if "parse_kernel_string(kernel)" not in body or "request.version_label" not in body:
                raise PlatformError("installed CLI cannot prove exact-version output/status support")
        if ("_is_within_directory(target_root, os.path.realpath(outfile))" not in functions["kernels_output"]
                or "os.path.commonpath" not in functions.get("_is_within_directory", "")):
            raise PlatformError("installed CLI lacks inspected output path containment checks")
        if "page_token" not in functions.get("kernels_output_cli", ""):
            raise PlatformError("installed CLI lacks inspected exact-version pagination support")
        if "while token and page_token is None" not in functions["kernels_output"]:
            raise PlatformError("installed CLI cannot bound explicit output pagination")
        if "return owner, slug, version" not in functions.get("parse_kernel_string", ""):
            raise PlatformError("installed CLI version parsing is unsupported")
        return {"cli_version": distribution.version, "source_sha256": digest(source.encode()),
                "identity_check": "installed official CLI version-label plumbing"}
    except (importlib.metadata.PackageNotFoundError, OSError, SyntaxError):
        raise PlatformError("exact-version capability unavailable; upgrade and inspect the official CLI") from None


def kernel_status(cli, kernel, version, timeout=None):
    identifier = exact_kernel(kernel, version)
    output = cli.run(["kernels", "status", identifier], timeout=timeout)
    match = re.search(re.escape(identifier) + r' has status ["\']?([A-Za-z_]+)', output)
    if not match:
        raise PlatformError("unrecognized kernel status response")
    state = match.group(1).lower()
    if state in ("error", "failed", "cancelled", "canceled"):
        raise PlatformError("kernel terminated unsuccessfully")
    if state not in ("complete", "running", "queued", "pending"):
        raise PlatformError("unknown kernel state")
    return {"kernel": kernel, "version": version, "status": state}


def wait_kernel(cli, kernel, version, timeout=3600, interval=5, max_interval=60,
                clock=time.monotonic, sleep=time.sleep, report=None):
    if any(not math.isfinite(x) or x <= 0 for x in (timeout, interval, max_interval)):
        raise PlatformError("poll durations must be positive and finite")
    deadline, delay, polls = clock() + timeout, min(interval, max_interval), 0
    while clock() < deadline:
        result = kernel_status(cli, kernel, version, timeout=max(.001, deadline-clock()))
        polls += 1
        if report:
            report({"poll": polls, "status": result["status"]})
        if result["status"] == "complete":
            return {**result, "polls": polls}
        remaining = deadline - clock()
        if remaining > 0:
            sleep(min(delay, remaining))
        delay = min(delay * 1.7, max_interval)
    raise PlatformError("bounded kernel wait timed out; resume status instead of pushing again")


def safe_extract_zip(archive, destination, max_bytes=10_000_000_000):
    destination = Path(destination)
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        if sum(item.file_size for item in members) > max_bytes:
            raise PlatformError("archive exceeds extraction budget")
        seen = set()
        for item in members:
            name = PurePosixPath(item.filename)
            mode = item.external_attr >> 16
            if (name.is_absolute() or ".." in name.parts or "\\" in item.filename
                    or name.parts and ":" in name.parts[0] or stat.S_ISLNK(mode)
                    or item.filename in seen):
                raise PlatformError("unsafe archive member")
            seen.add(item.filename)
            target = destination.joinpath(*name.parts)
            if not target.resolve().is_relative_to(destination.resolve()) or target.exists():
                raise PlatformError("archive would escape or overwrite destination")
        destination.mkdir(parents=True, exist_ok=True)
        for item in members:
            target = destination.joinpath(*PurePosixPath(item.filename).parts)
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(item) as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst)


def fetch_output(cli, kernel, version, out, capability=None):
    identity = capability() if capability else exact_version_capability(cli.executable)
    identifier = exact_kernel(kernel, version)
    status = kernel_status(cli, kernel, version)
    if status["status"] != "complete":
        raise PlatformError("output requires a completed exact kernel version")
    out = Path(out)
    if out.exists():
        raise PlatformError("output destination must be new")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kernel-output-", dir=out.parent) as temporary:
        download = Path(temporary) / "files"
        download.mkdir()
        token, seen, pages = None, set(), 0
        for page in range(1, 101):
            page_dir = Path(temporary) / f"page-{page}"
            page_dir.mkdir()
            arguments = ["kernels", "output", identifier, "-p", page_dir,
                         "--page-token", token or ""]
            response = cli.run(arguments)
            pages += 1
            for item in manifest(page_dir, upload=False):
                relative = item["file"]
                if relative == "output-receipt.json":
                    raise PlatformError("remote output uses a reserved receipt filename")
                target = download / relative
                if target.exists():
                    if not target.is_file() or file_sha(target) != item["sha256"]:
                        raise PlatformError("output pages contain conflicting duplicate files")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(page_dir / relative, target)
            tokens = re.findall(r"^Next page token:\s*(\S+)\s*$", response, re.MULTILINE)
            if not tokens:
                break
            if len(tokens) != 1 or tokens[0] in seen:
                raise PlatformError("output pagination repeated or returned ambiguous token")
            token = tokens[0]
            seen.add(token)
        else:
            raise PlatformError("output exceeded the 100-page download budget")
        files = manifest(download, upload=False)
        # No extraction is delegated to CLI: its output command downloads individual files.
        receipt = {"kernel": kernel, "version": version, "status": "downloaded",
            "files": files, "cli": identity, "exact_identifier": identifier, "download_pages": pages}
        atomic_json(download / "output-receipt.json", receipt, exclusive=True)
        os.rename(download, out)
    return receipt


def utc_now():
    return datetime.now(timezone.utc)


def approval_for(path, request, receipts, now=None):
    approval = read_json(path)
    if approval.get("request") != request:
        raise PlatformError("approval must match the complete request exactly")
    try:
        expiry = datetime.fromisoformat(approval["expires_at"].replace("Z", "+00:00"))
        maximum = approval["quota"]["max_operations"]
        if expiry.tzinfo is None or expiry <= (now or utc_now()):
            raise PlatformError("approval expired or lacks timezone")
        if type(maximum) is not int or maximum < 1:
            raise PlatformError("approval quota must be a positive integer")
    except (KeyError, ValueError, TypeError):
        raise PlatformError("invalid approval expiry/quota") from None
    approval_sha = digest(canonical(approval))
    used = sum(item.get("approval_sha256") == approval_sha for item in receipts)
    if used >= maximum:
        raise PlatformError("approval quota exhausted; unknown attempts consume quota")
    return approval_sha


class Ledger:
    def __init__(self, folder):
        self.folder = Path(folder)

    def receipts(self):
        return [read_json(path) for path in self.folder.glob("*.json")]

    def reserve(self, request, approval):
        self.folder.mkdir(parents=True, exist_ok=True)
        lock = self.folder / ".reserve-lock"
        try:
            lock.mkdir()
        except FileExistsError:
            raise PlatformError("receipt ledger locked; inspect pending attempts before recovery") from None
        try:
            receipts = self.receipts()
            # Messages do not turn identical bytes or versions into a new submission.
            if request["operation"] == "submit-file":
                identity = {key: request[key] for key in ("operation", "competition", "artifact_sha256")}
            elif request["operation"] == "submit-code":
                identity = {key: request[key] for key in ("operation", "competition", "kernel", "version", "output_file")}
            else:
                identity = {key: value for key, value in request.items() if key != "message"}
            identity_sha = digest(canonical(identity))
            if any(item.get("identity_sha256") == identity_sha for item in receipts):
                raise PlatformError("duplicate attempt exists; use reconcile, never blind resubmit")
            approval_sha = approval_for(approval, request, receipts)
            attempt = digest(canonical(request))
            value = {"attempt_id": attempt, "identity_sha256": identity_sha,
                "request": request, "approval_sha256": approval_sha,
                "state": "pending", "created_at": utc_now().isoformat()}
            atomic_json(self.folder / f"{attempt}.json", value, exclusive=True)
            return value
        finally:
            lock.rmdir()

    def update(self, receipt, **changes):
        receipt = {**receipt, **changes, "updated_at": utc_now().isoformat()}
        atomic_json(self.folder / f'{receipt["attempt_id"]}.json', receipt)
        return receipt


@contextmanager
def upload_snapshot(arguments, request):
    """Hash the bytes the CLI will upload, not a mutable caller-owned path."""
    arguments = list(arguments)
    with tempfile.TemporaryDirectory(prefix="kaggle-approved-upload-") as temporary:
        root = Path(temporary)
        if request["operation"] == "submit-file":
            index = arguments.index("-f") + 1
            target = root / request["file_name"]
            if target.name != request["file_name"] or not request["file_name"]:
                raise PlatformError("invalid submission filename")
            shutil.copyfile(arguments[index], target)
            target.chmod(0o400)
            if file_sha(target) != request["artifact_sha256"]:
                raise PlatformError("submission bytes changed before upload snapshot")
            arguments[index] = target
        elif request["operation"] in ("kernel-push", "dataset-create", "dataset-version"):
            index = arguments.index("-p") + 1
            target = root / "package"
            shutil.copytree(arguments[index], target, symlinks=True)
            if package_identity(target) != request["package_sha256"]:
                raise PlatformError("upload package changed before snapshot")
            metadata = read_json(target / ("kernel-metadata.json" if request["operation"] == "kernel-push" else "dataset-metadata.json"))
            if metadata.get("id") != request["id"]:
                raise PlatformError("upload snapshot identifier differs from approved target")
            arguments[index] = target
        yield arguments


def execute_mutation(cli, arguments, request, ledger, approval):
    with upload_snapshot(arguments, request) as frozen:
        return _execute_frozen(cli, frozen, request, ledger, approval)


def _execute_frozen(cli, arguments, request, ledger, approval):
    receipt = ledger.reserve(request, approval)
    try:
        output = cli.run(arguments)
        if request["operation"].startswith("submit-"):
            match = re.search(r"Submission ref:\s*(\d+)", output)
            if not match:
                raise PlatformError("submission response lacks reference; reconcile before any retry")
            return ledger.update(receipt, state="submitted", submission_ref=match.group(1))
        if request["operation"] == "kernel-push":
            match = re.search(r"[Vv]ersion\s+(\d+)", output)
            if not match:
                raise PlatformError("push response lacks exact version; inspect platform history")
            return ledger.update(receipt, state="pushed", version=int(match.group(1)))
        marker = ("Your private Dataset is being created." if request["operation"] == "dataset-create"
                  else "Dataset version is being created.")
        if marker not in output:
            raise PlatformError("dataset response lacks a recognized acceptance marker; inspect platform history")
        return ledger.update(receipt, state="uploaded", evidence="CLI success; remote privacy/version not yet read back")
    except BaseException:
        ledger.update(receipt, state="unknown", reason="network or response uncertainty; do not repeat mutation")
        raise


def parse_submissions(output):
    try:
        parsed = json.loads(output)
        if isinstance(parsed, dict):
            parsed = parsed.get("submissions", parsed.get("items", [parsed]))
        if not isinstance(parsed, list) or any(not isinstance(item, dict) for item in parsed):
            raise ValueError()
        return parsed
    except (json.JSONDecodeError, ValueError):
        reader = csv.DictReader(io.StringIO(output))
        if not reader.fieldnames or "ref" not in reader.fieldnames:
            raise PlatformError("submission query must return JSON or CSV including ref") from None
        return list(reader)


def reconcile(cli, ledger, attempt, remote_ref=None):
    if not re.fullmatch(r"[a-f0-9]{64}", attempt):
        raise PlatformError("invalid attempt id")
    receipt = read_json(ledger.folder / f"{attempt}.json")
    request = receipt["request"]
    if not request["operation"].startswith("submit-"):
        raise PlatformError("this reconcile adapter supports submissions only; query exact kernel status separately")
    rows = parse_submissions(cli.run(["competitions", "submissions", "-c", request["competition"], "--csv"]))
    reference = remote_ref or receipt.get("submission_ref")
    if reference:
        matches = [row for row in rows if str(row.get("ref")) == str(reference)]
    else:
        matches = [row for row in rows if row.get("description") == request["message"]]
    if len(matches) != 1:
        raise PlatformError("submission identity unresolved; no automatic resubmission is allowed")
    row = matches[0]
    if str(row.get("description", "")) != request["message"]:
        raise PlatformError("remote description differs from approved message")
    if request["operation"] == "submit-file" and row.get("fileName") != request["file_name"]:
        raise PlatformError("remote file name differs; identity unresolved")
    if not row.get("ref"):
        raise PlatformError("remote submission lacks reference")
    # Description alone cannot prove the identity of a timed-out upload.
    if not reference:
        return ledger.update(receipt, state="unknown", candidate_ref=str(row["ref"]),
            reason="candidate found; explicitly verify remote artifact before binding --remote-ref")
    status = str(row.get("status", "")).lower()
    score = row.get("publicScore", row.get("public_score"))
    if status in ("error", "failed", "cancelled", "canceled"):
        ledger.update(receipt, state="failed", submission_ref=str(row["ref"]), remote_status=status)
        raise PlatformError("remote submission failed")
    if status == "complete":
        try:
            number = float(score)
            if not math.isfinite(number):
                raise ValueError()
        except (TypeError, ValueError):
            ledger.update(receipt, state="score_unavailable", submission_ref=str(row["ref"]), remote_status=status)
            raise PlatformError("completed submission has blank or invalid score; score is unverified") from None
        return ledger.update(receipt, state="scored", submission_ref=str(row["ref"]), public_score=number)
    if status not in ("pending", "running", "submitted", "submitting"):
        raise PlatformError("unknown remote submission status")
    return ledger.update(receipt, state="submitted", submission_ref=str(row["ref"]), remote_status=status)


def code_receipt(path, kernel, version, output_file, output_dir):
    receipt = read_json(path)
    exact_kernel(kernel, version)
    if (receipt.get("kernel") != kernel or receipt.get("version") != version
            or receipt.get("validation") != "PASS"
            or receipt.get("output_file") != output_file
            or not re.fullmatch(r"[a-f0-9]{64}", receipt.get("artifact_sha256", ""))
            or not re.fullmatch(r"[a-f0-9]{64}", receipt.get("output_receipt_sha256", ""))):
        raise PlatformError("code submission needs a validated exact-version receipt")
    remote = read_json(Path(output_dir) / "output-receipt.json")
    if (remote.get("kernel") != kernel or remote.get("version") != version
            or digest(canonical(remote)) != receipt["output_receipt_sha256"]):
        raise PlatformError("download receipt changed or has a different version")
    relative = PurePosixPath(output_file)
    artifact = Path(output_dir) / output_file
    expected = [item for item in remote.get("files", []) if item.get("file") == output_file]
    if (relative.is_absolute() or ".." in relative.parts or "\\" in output_file
            or artifact.is_symlink() or not artifact.resolve().is_relative_to(Path(output_dir).resolve())
            or len(expected) != 1 or expected[0].get("sha256") != receipt["artifact_sha256"]
            or file_sha(artifact) != receipt["artifact_sha256"]):
        raise PlatformError("validated output bytes changed")
    return digest(canonical(receipt))


def validate_output(output_dir, kernel, version, file_name, sample, ids, numbers, receipt_path):
    root = Path(output_dir)
    receipt_path = Path(receipt_path)
    remote = read_json(root / "output-receipt.json")
    exact_kernel(kernel, version)
    if remote.get("kernel") != kernel or remote.get("version") != version:
        raise PlatformError("downloaded output belongs to a different kernel/version")
    relative = PurePosixPath(file_name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in file_name:
        raise PlatformError("invalid output file name")
    artifact = root / file_name
    if artifact.is_symlink() or not artifact.resolve().is_relative_to(root.resolve()):
        raise PlatformError("unsafe output artifact")
    expected = [item for item in remote["files"] if item["file"] == file_name]
    with tempfile.TemporaryDirectory(prefix="kaggle-output-validation-") as temporary:
        frozen = Path(temporary) / "artifact.csv"
        shutil.copyfile(artifact, frozen)
        frozen.chmod(0o400)
        if len(expected) != 1 or file_sha(frozen) != expected[0]["sha256"]:
            raise PlatformError("output artifact hash differs from download receipt")
        checked = check_csv(Path(sample), frozen, ids, numbers)
        if checked["sha256"] != expected[0]["sha256"]:
            raise PlatformError("output snapshot changed during validation")
    result = {"validation": "PASS", "kernel": kernel, "version": version,
        "output_file": file_name, "artifact_sha256": checked["sha256"],
        "output_receipt_sha256": digest(canonical(remote)), "csv_check": checked,
        "scope": "CSV structure and downloaded output identity; competition semantics require separate review"}
    atomic_json(receipt_path, result, exclusive=True)
    return result


def intel_arguments(kind, competition=None, query=None, kernel=None, out=None):
    if competition:
        safe_slug(competition)
    if kind == "search":
        return ["competitions", "list", "--csv", "-s", query or ""]
    if kind == "version":
        return ["--version"]
    if kind == "quota":
        return ["quota", "--csv"]
    if kind == "submission-limits":
        if not competition:
            raise PlatformError("competition is required")
        return ["competitions", "submission-limits", "-c", competition, "--json"]
    if kind == "files":
        if not competition:
            raise PlatformError("competition is required")
        return ["competitions", "files", "-c", competition, "--csv"]
    if kind == "leaderboard":
        if not competition:
            raise PlatformError("competition is required")
        return ["competitions", "leaderboard", "-c", competition, "--show", "--csv"]
    if kind == "kernels":
        if not competition:
            raise PlatformError("competition is required")
        return ["kernels", "list", "--competition", competition, "--csv"]
    if kind == "pull":
        if not kernel or not out:
            raise PlatformError("pull needs kernel and a new output directory")
        safe_slug(kernel, paired=True)
        return ["kernels", "pull", kernel, "-p", str(out), "--metadata"]
    raise PlatformError("unsupported intelligence operation")


class MCP:
    """Minimal Streamable HTTP JSON-RPC client; no guessed tool names."""
    def __init__(self, url, token_env=None, timeout=30, transport=None):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise PlatformError("MCP endpoint must be HTTPS without credentials or query")
        self.url, self.timeout = url, timeout
        self.token = os.environ.get(token_env, "") if token_env else ""
        self.session, self.counter, self.transport = None, 0, transport or self._transport

    def _transport(self, payload, headers):
        request = urllib.request.Request(self.url, data=canonical(payload), headers=headers, method="POST")
        # Never forward a bearer token across redirects.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        try:
            with urllib.request.build_opener(NoRedirect).open(request, timeout=self.timeout) as response:
                if "id" not in payload and response.status in (200, 202, 204):
                    return {}, response.headers.get("Mcp-Session-Id")
                content_type = response.headers.get("Content-Type", "")
                if "text/event-stream" in content_type:
                    return self._sse_result(response, payload.get("id")), response.headers.get("Mcp-Session-Id")
                if "application/json" not in content_type:
                    raise PlatformError("unsupported MCP response content type")
                raw = response.read(10_000_001)
                if len(raw) > 10_000_000:
                    raise PlatformError("MCP response exceeds limit")
                return json.loads(raw), response.headers.get("Mcp-Session-Id")
        except (OSError, ValueError):
            raise PlatformError("MCP transport failed; sensitive endpoint details withheld") from None

    def _sse_result(self, response, request_id):
        deadline, used, data = time.monotonic() + self.timeout, 0, []
        while time.monotonic() < deadline:
            line = response.readline(min(65536, 10_000_001 - used))
            used += len(line)
            if used > 10_000_000 or len(line) == 65536 and not line.endswith(b"\n"):
                raise PlatformError("MCP event stream exceeds response budget")
            if not line or line in (b"\n", b"\r\n"):
                if data:
                    event = json.loads("\n".join(data))
                    if isinstance(event, dict) and event.get("id") == request_id:
                        return event
                    data = []
                if not line:
                    break
            elif line.startswith(b"data:"):
                data.append(line[5:].decode("utf-8").strip())
        raise PlatformError("MCP event stream ended or timed out without matching JSON-RPC result")

    def rpc(self, method, params=None, notification=False):
        self.counter += 1
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if not notification:
            payload["id"] = self.counter
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        if self.session:
            headers["Mcp-Session-Id"] = self.session
            headers["MCP-Protocol-Version"] = "2025-03-26"
        result, session = self.transport(payload, headers)
        self.session = session or self.session
        if notification:
            return None
        if (not isinstance(result, dict) or result.get("jsonrpc") != "2.0"
                or result.get("id") != self.counter or "error" in result or "result" not in result):
            raise PlatformError("MCP protocol returned an error or mismatched response")
        return result["result"]

    def initialize(self):
        result = self.rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
            "clientInfo": {"name": "kaggle-local-tools", "version": "1"}})
        if result.get("protocolVersion") != "2025-03-26":
            raise PlatformError("MCP protocol version is unsupported")
        self.rpc("notifications/initialized", notification=True)

    def tools(self):
        result, cursor = [], None
        for _ in range(20):
            page = self.rpc("tools/list", {"cursor": cursor} if cursor else {})
            result.extend(page.get("tools", []))
            cursor = page.get("nextCursor")
            if not cursor:
                return result
        raise PlatformError("MCP tool listing exceeded page budget")

    def call_readonly(self, name, arguments):
        listed = [item for item in self.tools() if item.get("name") == name]
        if len(listed) != 1 or listed[0].get("annotations", {}).get("readOnlyHint") is not True:
            raise PlatformError("MCP tool must be discovered and explicitly annotated read-only")
        result = self.rpc("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            raise PlatformError("MCP tool returned an error")
        return result


def permissions(parser, approval=False):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--execute", action="store_true")
    group.add_argument("--dry-run", action="store_true")
    if approval:
        parser.add_argument("--approval", type=Path)
        parser.add_argument("--ledger", type=Path, default=Path("platform-receipts"))


def kernel_args(parser):
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--version", required=True, type=int)


def csv_args(parser):
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--id-column", action="append", required=True)
    parser.add_argument("--numeric-column", action="append", required=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", default="kaggle")
    parser.add_argument("--cli-timeout", type=float, default=60)
    commands = parser.add_subparsers(dest="command", required=True)
    credentials = commands.add_parser("credentials")
    credentials.add_argument("--config-dir", type=Path)
    credentials.add_argument("--fix-perms", action="store_true")
    permissions(credentials)
    for name in ("prepare-kernel", "prepare-dataset"):
        prepare = commands.add_parser(name)
        prepare.add_argument("source", type=Path)
        prepare.add_argument("--out", type=Path, required=True)
        prepare.add_argument("--id", required=True)
        prepare.add_argument("--title", required=True)
        permissions(prepare)
        if name == "prepare-kernel":
            prepare.add_argument("--competition")
            prepare.add_argument("--gpu", action="store_true")
            prepare.add_argument("--internet", action="store_true")
            prepare.add_argument("--dataset", action="append", default=[])
        else:
            prepare.add_argument("--license", required=True)
    for name in ("kernel-push", "dataset-create", "dataset-version"):
        push = commands.add_parser(name)
        push.add_argument("folder", type=Path)
        push.add_argument("--message", default="")
        permissions(push, approval=True)
    for name in ("kernel-status", "kernel-wait", "kernel-output"):
        remote = commands.add_parser(name)
        kernel_args(remote)
        permissions(remote)
        if name == "kernel-wait":
            remote.add_argument("--timeout", type=float, default=3600)
            remote.add_argument("--interval", type=float, default=5)
            remote.add_argument("--max-interval", type=float, default=60)
        if name == "kernel-output":
            remote.add_argument("--out", type=Path, required=True)
    validation = commands.add_parser("validate-output")
    validation.add_argument("folder", type=Path)
    kernel_args(validation)
    validation.add_argument("--file", required=True)
    validation.add_argument("--receipt", type=Path, required=True)
    csv_args(validation)
    permissions(validation)
    for name in ("submit-file", "submit-code"):
        submit = commands.add_parser(name)
        submit.add_argument("--competition", required=True)
        submit.add_argument("--message", required=True)
        submit.add_argument("--file", required=True)
        permissions(submit, approval=True)
        if name == "submit-file":
            csv_args(submit)
        else:
            kernel_args(submit)
            submit.add_argument("--validated-receipt", type=Path, required=True)
            submit.add_argument("--output-dir", type=Path, required=True)
    resume = commands.add_parser("reconcile")
    resume.add_argument("--ledger", type=Path, required=True)
    resume.add_argument("--attempt", required=True)
    resume.add_argument("--remote-ref")
    permissions(resume)
    intel = commands.add_parser("intel")
    intel.add_argument("kind", choices=("search", "files", "leaderboard", "kernels", "pull", "version", "quota", "submission-limits"))
    intel.add_argument("--competition")
    intel.add_argument("--query")
    intel.add_argument("--kernel")
    intel.add_argument("--out", type=Path)
    permissions(intel)
    mcp = commands.add_parser("mcp")
    mcp.add_argument("--endpoint", required=True)
    mcp.add_argument("--token-env")
    mcp.add_argument("--tool")
    mcp.add_argument("--arguments", type=Path)
    mcp.add_argument("--out", type=Path, required=True)
    permissions(mcp)
    return parser


def dispatch(args):
    command = args.command
    if command == "credentials":
        result = credential_check(args.config_dir, fix_perms=args.fix_perms and args.execute)
        if args.fix_perms:
            result["permission_repair"] = "executed" if args.execute else "dry_run"
        return result
    if command.startswith("prepare-"):
        if not args.execute:
            return {"dry_run": True, "operation": command, "private": True}
        if command == "prepare-kernel":
            return prepare_kernel(args.source, args.out, args.id, args.title,
                args.competition, args.gpu, args.internet, args.dataset)
        return prepare_dataset(args.source, args.out, args.id, args.title, args.license)
    cli = CLI(args.cli, args.cli_timeout)
    request, arguments = None, None
    if command in ("kernel-push", "dataset-create", "dataset-version"):
        sha = package_identity(args.folder)
        metadata = read_json(args.folder / ("kernel-metadata.json" if command == "kernel-push" else "dataset-metadata.json"))
        safe_slug(metadata["id"], paired=True)
        if command == "kernel-push" and metadata.get("is_private") is not True:
            raise PlatformError("only private kernel pushes are supported")
        request = {"operation": command, "id": metadata["id"], "package_sha256": sha,
                   "private": True, "message": args.message}
        if command == "kernel-push":
            arguments = ["kernels", "push", "-p", args.folder]
        elif command == "dataset-create":
            arguments = ["datasets", "create", "-p", args.folder, "--dir-mode", "zip"]
        else:
            # Dataset versions inherit existing privacy: do not claim remote privacy is verified.
            request["private"] = "inherited-unverified"
            arguments = ["datasets", "version", "-p", args.folder, "-m", args.message, "--dir-mode", "zip"]
    elif command == "submit-file":
        safe_slug(args.competition)
        checked = check_csv(args.sample, Path(args.file), args.id_column, args.numeric_column)
        request = {"operation": command, "competition": args.competition,
            "artifact_sha256": checked["sha256"], "file_name": Path(args.file).name,
            "message": args.message}
        arguments = ["competitions", "submit", "-c", args.competition, "-f", args.file, "-m", args.message]
    elif command == "submit-code":
        safe_slug(args.competition)
        exact_kernel(args.kernel, args.version)
        sha = code_receipt(args.validated_receipt, args.kernel, args.version, args.file, args.output_dir)
        request = {"operation": command, "competition": args.competition,
            "kernel": args.kernel, "version": args.version, "output_file": args.file,
            "validated_receipt_sha256": sha, "message": args.message}
        arguments = ["competitions", "submit", "-c", args.competition, "-k", args.kernel,
                     "-v", str(args.version), "-f", args.file, "-m", args.message]
    if request:
        if not args.execute:
            return {"dry_run": True, "request": request, "approval_required": True}
        if not args.approval:
            raise PlatformError("execution requires a matching user-approved JSON file")
        return execute_mutation(cli, arguments, request, Ledger(args.ledger), args.approval)
    if not args.execute:
        return {"dry_run": True, "operation": command, "network_called": False}
    if command.startswith("kernel-"):
        exact_version_capability(cli.executable)
        if command == "kernel-status":
            return kernel_status(cli, args.kernel, args.version)
        if command == "kernel-wait":
            return wait_kernel(cli, args.kernel, args.version, args.timeout, args.interval,
                args.max_interval, report=lambda value: print(json.dumps(value), flush=True))
        return fetch_output(cli, args.kernel, args.version, args.out)
    if command == "validate-output":
        return validate_output(args.folder, args.kernel, args.version, args.file,
            args.sample, args.id_column, args.numeric_column, args.receipt)
    if command == "reconcile":
        return reconcile(cli, Ledger(args.ledger), args.attempt, args.remote_ref)
    if command == "intel":
        arguments = intel_arguments(args.kind, args.competition, args.query, args.kernel, args.out)
        if args.kind == "pull":
            if args.out.exists():
                raise PlatformError("pull destination must be new")
            cli.run(arguments)
            return {"pulled": True, "source_identity": "latest; not exact-version output evidence",
                    "files": manifest(args.out, upload=False)}
        # Save remote text locally; do not echo untrusted text or endpoint details into logs.
        if not args.out:
            raise PlatformError("intel requires --out to preserve local evidence")
        if args.out.exists():
            raise PlatformError("intel evidence destination must be new")
        output = cli.run(arguments)
        with args.out.open("x", encoding="utf-8") as handle:
            handle.write(output)
        return {"intel_saved": True, "sha256": digest(output.encode())}
    if command == "mcp":
        if args.out.exists():
            raise PlatformError("MCP evidence destination must be new")
        client = MCP(args.endpoint, args.token_env)
        client.initialize()
        result = client.call_readonly(args.tool, read_json(args.arguments) if args.arguments else {}) if args.tool else client.tools()
        atomic_json(args.out, result, exclusive=True)
        return {"mcp_evidence_saved": True, "sha256": digest(canonical(result)),
                "note": "readOnlyHint is advisory; trust and review the configured server"}
    raise PlatformError("unsupported command")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = dispatch(args)
    except PlatformError as error:
        parser.exit(2, str(error) + "\n")
    except (OSError, ValueError, KeyError, TypeError, csv.Error, SyntaxError):
        parser.exit(2, "invalid local input or artifact; details withheld to protect paths and credentials\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
