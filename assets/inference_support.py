"""Model-independent building blocks for observable, bounded inference runs."""

import csv
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path


def verify_artifacts(root, entries):
    root = Path(root).resolve()
    checked = []
    for entry in entries:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("artifact absent or outside artifact root")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError("artifact digest mismatch")
        checked.append(entry["path"])
    if not checked:
        raise ValueError("empty artifact manifest")
    return checked


class RunBudget:
    def __init__(self, seconds, reserve=0, clock=time.monotonic):
        if not math.isfinite(seconds) or not math.isfinite(reserve) or not 0 <= reserve < seconds:
            raise ValueError("require 0 <= reserve < finite time budget")
        self.clock = clock
        self.end = clock() + seconds
        self.reserve = reserve

    def remaining(self):
        return max(0, self.end - self.clock())

    def optional_pass_fits(self, estimated_seconds):
        if not math.isfinite(estimated_seconds) or estimated_seconds < 0:
            raise ValueError("invalid optional-pass estimate")
        return self.remaining() >= estimated_seconds + self.reserve

    def require(self, estimated_seconds=0):
        if not self.optional_pass_fits(estimated_seconds):
            raise TimeoutError("insufficient budget; do not replace real predictions with a template")


def predict_items(items, predict, report_path, budget=None, recoverable=(), fallback=None):
    """Only declared item-level errors may use an explicitly supplied fallback."""
    if recoverable and fallback is None:
        raise ValueError("recoverable failures need an explicit, validated fallback")
    reports = []
    with Path(report_path).open("x", encoding="utf-8") as report:
        for item in items:
            if budget:
                budget.require()
            start = time.monotonic()
            status = "predicted"
            try:
                value = predict(item)
            except recoverable as exc:
                value = fallback(item, exc)
                status = "fallback"
            elapsed = time.monotonic() - start
            report.write(json.dumps({"item_index": len(reports), "seconds": elapsed, "status": status}) + "\n")
            report.flush()
            reports.append(value)
    return reports


def atomic_csv(path, columns, rows, numeric_columns=()):
    path = Path(path)
    if len(set(columns)) != len(columns) or not columns:
        raise ValueError("unique output columns required")
    if not set(numeric_columns) <= set(columns):
        raise ValueError("unknown numeric columns")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".prediction-", suffix=".csv", dir=path.parent)
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                if set(row) != set(columns) or any(value is None or not str(value).strip() for value in row.values()):
                    raise ValueError("prediction row does not match output schema")
                if any(not math.isfinite(float(row[name])) for name in numeric_columns):
                    raise ValueError("non-finite prediction")
                writer.writerow(row)
                count += 1
            if not count:
                raise ValueError("no predictions produced")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"rows": count, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
