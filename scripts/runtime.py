"""Measure local execution and project a bounded remote inference plan."""

import argparse
import csv
import json
import math
import resource
import runpy
import time
from pathlib import Path


def p95(values):
    ordered = sorted(values)
    if not ordered or any(not math.isfinite(x) or x < 0 for x in ordered):
        raise ValueError("latency samples must be finite and nonnegative")
    return ordered[math.ceil(.95 * len(ordered)) - 1]


def read_timings(path):
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not {"seconds", "passes", "fold", "tta"}.issubset(reader.fieldnames or []):
            raise ValueError("timings need seconds, passes, fold and tta")
        rows = []
        for row in reader:
            parsed = dict(row)
            for key in ("seconds", "passes", "decode_seconds", "rss_mb", "gpu_mb", "unit_index"):
                if row.get(key, "") != "":
                    parsed[key] = float(row[key])
                    if not math.isfinite(parsed[key]) or parsed[key] < 0:
                        raise ValueError("invalid measurement")
            if parsed["passes"] <= 0 or parsed["passes"] != int(parsed["passes"]):
                raise ValueError("passes must be a positive integer")
            if not row["fold"] or not row["tta"]:
                raise ValueError("fold and tta identities are required")
            rows.append(parsed)
    if not rows:
        raise ValueError("empty timing table")
    return rows


def memory_projection(rows, field, hidden_items, limit=None):
    samples = [(row.get("unit_index"), row[field]) for row in rows if isinstance(row.get(field), (int, float))]
    if not samples:
        return {"measured": False, "within_budget": None}
    peak = max(value for _, value in samples)
    by_unit = {}
    for index, value in samples:
        if index is not None:
            by_unit[index] = max(value, by_unit.get(index, 0))
    ordered = sorted(by_unit.items())
    slopes = [max(0, (b[1]-a[1])/(b[0]-a[0])) for a, b in zip(ordered, ordered[1:]) if b[0] > a[0]]
    growth = max(slopes, default=0)
    projected = peak + growth * max(0, hidden_items - max(by_unit, default=hidden_items))
    return {"measured": True, "observed_peak_mb": peak, "growth_mb_per_unit": growth,
            "projected_peak_mb": projected, "limit_mb": limit,
            "growth_measured": len(ordered) >= 2,
            "within_budget": None if limit is None else projected <= limit,
            "assumption": "conservative observed linear growth; nonlinear leaks remain unverified"}


def project_csv(path, hidden_items, folds, tta, loop_order="unit-outer",
                startup_seconds=0, finalize_seconds=0, queue_seconds=0,
                crash_min_seconds=0, crash_max_seconds=0, safety_factor=1.25,
                budget_seconds=None, rss_limit_mb=None, gpu_limit_mb=None, config=None):
    rows = read_timings(path)
    if config is not None:
        rows = [row for row in rows if row.get("config") == config]
        if not rows:
            raise ValueError("no measurements for requested config")
    if (loop_order not in ("unit-outer", "fold-outer") or not folds or not tta
            or len(set(map(str, folds))) != len(folds) or len(set(map(str, tta))) != len(tta)):
        raise ValueError("invalid fold/TTA plan")
    if type(hidden_items) is not int or hidden_items < 1:
        raise ValueError("hidden items must be positive")
    costs = [startup_seconds, finalize_seconds, queue_seconds, crash_min_seconds, crash_max_seconds, safety_factor]
    if any(not math.isfinite(x) or x < 0 for x in costs) or crash_max_seconds < crash_min_seconds or safety_factor < 1:
        raise ValueError("invalid overhead or crash interval")
    for limit in (budget_seconds, rss_limit_mb, gpu_limit_mb):
        if limit is not None and (not math.isfinite(limit) or limit <= 0):
            raise ValueError("budgets must be finite and positive")
    phases = []
    for fold in folds:
        for augmentation in tta:
            samples = [row["seconds"] / row["passes"] for row in rows
                       if row["fold"] == str(fold) and row["tta"] == str(augmentation)]
            if not samples:
                raise ValueError("unmeasured fold/TTA combination")
            phases.append({"fold": str(fold), "tta": str(augmentation),
                "p95_seconds_per_unit": p95(samples), "measurements": len(samples)})
    decodes = [row["decode_seconds"] for row in rows if isinstance(row.get("decode_seconds"), (float, int))]
    if not decodes:
        raise ValueError("decode_seconds measurements are required")
    decode = p95(decodes)
    decode_passes = 1 if loop_order == "unit-outer" else len(phases)
    inference = hidden_items * sum(phase["p95_seconds_per_unit"] for phase in phases)
    decoding = hidden_items * decode * decode_passes
    base = startup_seconds + finalize_seconds + inference + decoding
    low = (base + crash_min_seconds) * safety_factor
    high = (base + crash_max_seconds) * safety_factor
    return {"config": config, "hidden_items": hidden_items, "loop_order": loop_order,
        "phases": phases, "decode_p95_seconds": decode, "decode_passes": decode_passes,
        "inference_seconds": inference, "decode_seconds": decoding,
        "startup_seconds": startup_seconds, "finalize_seconds": finalize_seconds,
        "queue_seconds": queue_seconds, "crash_interval_seconds": [crash_min_seconds, crash_max_seconds],
        "projected_runtime_interval_seconds": [low, high],
        "projected_wall_interval_seconds": [low + queue_seconds, high + queue_seconds],
        "budget_seconds": budget_seconds, "within_runtime_budget": None if budget_seconds is None else high <= budget_seconds,
        "rss": memory_projection(rows, "rss_mb", hidden_items, rss_limit_mb),
        "gpu": memory_projection(rows, "gpu_mb", hidden_items, gpu_limit_mb),
        "evidence": "measured CSV projection; same hardware, unit-size and decoding assumptions must be checked"}


def benchmark(driver, items, configs, out):
    """Time user-supplied decode/infer callbacks; never import until explicit execution."""
    hooks = runpy.run_path(str(driver), run_name="runtime_driver")
    if not callable(hooks.get("decode")) or not callable(hooks.get("infer")):
        raise ValueError("driver must define decode(item, config), infer(decoded, fold, tta, config)")
    records, overheads = [], []
    for config in configs:
        name = config["name"]
        folds, augmentations = config["folds"], config["tta"]
        if not folds or not augmentations or config.get("passes", 1) < 1:
            raise ValueError("empty config or invalid passes")
        start = time.perf_counter()
        if callable(hooks.get("startup")):
            hooks["startup"](config)
        startup = time.perf_counter() - start
        for index, item in enumerate(items, 1):
            start = time.perf_counter()
            decoded = hooks["decode"](item, config)
            decode_seconds = time.perf_counter() - start
            for fold in folds:
                for augmentation in augmentations:
                    if callable(hooks.get("synchronize")):
                        hooks["synchronize"]()
                    start = time.perf_counter()
                    for _ in range(config.get("passes", 1)):
                        hooks["infer"](decoded, fold, augmentation, config)
                    if callable(hooks.get("synchronize")):
                        hooks["synchronize"]()
                    seconds = time.perf_counter() - start
                    # ru_maxrss is a conservative process lifetime peak, not current RSS.
                    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    import sys
                    rss_mb = rss / (1024*1024 if sys.platform == "darwin" else 1024)
                    gpu = hooks["gpu_memory_mb"]() if callable(hooks.get("gpu_memory_mb")) else ""
                    records.append({"config": name, "unit_index": index, "seconds": seconds,
                        "passes": config.get("passes", 1), "fold": fold, "tta": augmentation,
                        "decode_seconds": decode_seconds, "rss_mb": rss_mb, "gpu_mb": gpu})
        start = time.perf_counter()
        if callable(hooks.get("finalize")):
            hooks["finalize"](config)
        overheads.append({"config": name, "startup_seconds": startup,
                          "finalize_seconds": time.perf_counter() - start})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    with (out / "timings.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (out / "overheads.json").write_text(json.dumps(overheads, indent=2) + "\n")
    return {"measured_rows": len(records), "configurations": len(overheads),
            "timings_file": "timings.csv", "overheads_file": "overheads.json"}


def project(startup_seconds, batch_seconds, measured_items, hidden_items, folds=1,
            tta=1, budget_seconds=None, crash_seconds=0, safety_factor=1.25):
    values = [startup_seconds, *batch_seconds, crash_seconds, safety_factor]
    if not batch_seconds or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("measurements must be finite and nonnegative")
    if any(not isinstance(x, int) or x < 1 for x in (measured_items, hidden_items, folds, tta)):
        raise ValueError("item counts, folds and TTA must be positive integers")
    if safety_factor < 1 or budget_seconds is not None and (
            not math.isfinite(budget_seconds) or budget_seconds <= 0):
        raise ValueError("invalid safety factor or runtime budget")
    ordered = sorted(batch_seconds)
    p95 = ordered[math.ceil(.95 * len(ordered)) - 1]
    one_pass = p95 / measured_items * hidden_items
    total = (startup_seconds + one_pass * folds * tta + crash_seconds) * safety_factor
    return {"startup_seconds": startup_seconds, "p95_batch_seconds": p95,
            "measured_items_per_batch": measured_items, "batches_measured": len(ordered),
            "hidden_items": hidden_items, "folds": folds, "tta": tta,
            "inference_passes": folds * tta, "crash_allowance_seconds": crash_seconds,
            "safety_factor": safety_factor, "projected_seconds": total,
            "budget_seconds": budget_seconds,
            "within_budget": None if budget_seconds is None else total <= budget_seconds,
            "evidence": "local measured latency; hidden-size and hardware assumptions require verification"}


def run_measured(script, artifact):
    """Run only when explicitly requested; preserve a failure artifact and exception."""
    start = time.monotonic()
    status = "failed"
    try:
        runpy.run_path(str(script), run_name="__main__")
        status = "complete"
    finally:
        artifact.write_text(json.dumps({"status": status,
            "elapsed_seconds": time.monotonic() - start}, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("project")
    plan.add_argument("measurements", type=Path, help="JSON with measured startup_seconds and batch_seconds")
    plan.add_argument("--hidden-items", type=int, required=True)
    plan.add_argument("--folds", type=int, default=1)
    plan.add_argument("--tta", type=int, default=1)
    plan.add_argument("--budget-seconds", type=float)
    plan.add_argument("--crash-seconds", type=float, default=0)
    plan.add_argument("--safety-factor", type=float, default=1.25)
    run = commands.add_parser("measure")
    run.add_argument("script", type=Path)
    run.add_argument("--artifact", type=Path, required=True)
    run.add_argument("--execute", action="store_true")
    measured = commands.add_parser("benchmark")
    measured.add_argument("--driver", type=Path, required=True)
    measured.add_argument("--items", type=Path, required=True, help="JSON list of representative unit inputs")
    measured.add_argument("--grid", type=Path, required=True, help="JSON list of named folds/TTA configurations")
    measured.add_argument("--out", type=Path, required=True)
    measured.add_argument("--execute", action="store_true")
    measured.add_argument("--dry-run", action="store_true")
    table = commands.add_parser("plan-csv")
    table.add_argument("timings", type=Path)
    table.add_argument("--hidden-items", type=int, required=True)
    table.add_argument("--grid", type=Path, required=True)
    table.add_argument("--overheads", type=Path)
    table.add_argument("--queue-seconds", type=float, default=0)
    table.add_argument("--crash-min-seconds", type=float, default=0)
    table.add_argument("--crash-max-seconds", type=float, default=0)
    table.add_argument("--budget-seconds", type=float)
    table.add_argument("--rss-limit-mb", type=float)
    table.add_argument("--gpu-limit-mb", type=float)
    table.add_argument("--safety-factor", type=float, default=1.25)
    args = parser.parse_args(argv)
    try:
        if args.command == "benchmark":
            if not args.execute or args.dry_run:
                print(json.dumps({"dry_run": True, "operation": "local-benchmark"}))
                return
            items = json.loads(args.items.read_text())
            grid = json.loads(args.grid.read_text())
            if not items or not grid:
                raise ValueError("benchmark requires items and configs")
            print(json.dumps(benchmark(args.driver, items, grid, args.out), indent=2))
        elif args.command == "plan-csv":
            overheads = json.loads(args.overheads.read_text()) if args.overheads else []
            results = []
            for config in json.loads(args.grid.read_text()):
                overhead = next((entry for entry in overheads if entry["config"] == config["name"]), {})
                results.append(project_csv(args.timings, args.hidden_items, config["folds"], config["tta"],
                    config.get("loop_order", "unit-outer"), overhead.get("startup_seconds", config.get("startup_seconds", 0)),
                    overhead.get("finalize_seconds", config.get("finalize_seconds", 0)), args.queue_seconds,
                    args.crash_min_seconds, args.crash_max_seconds, args.safety_factor,
                    args.budget_seconds, args.rss_limit_mb, args.gpu_limit_mb, config["name"]))
            print(json.dumps(results, indent=2))
        elif args.command == "measure":
            if not args.execute:
                print(json.dumps({"dry_run": True, "operation": "local-execution"}))
                return
            run_measured(args.script, args.artifact)
        else:
            data = json.loads(args.measurements.read_text())
            print(json.dumps(project(data["startup_seconds"], data["batch_seconds"],
                data["measured_items_per_batch"], args.hidden_items, args.folds, args.tta,
                args.budget_seconds, args.crash_seconds, args.safety_factor), indent=2))
    except (ValueError, OSError, KeyError, TypeError):
        parser.exit(2, "runtime input or execution failed; inspect local artifacts\n")


if __name__ == "__main__":
    main()
