"""Compare CSV unit,fold,score files; positive gains always favor the candidate."""

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path


def read_scores(path):
    result = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["unit", "fold", "score"]:
            raise ValueError("score columns must be unit,fold,score")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("score row has an incorrect number of fields")
            unit, fold = row["unit"], row["fold"]
            if not unit or not fold or unit in result:
                raise ValueError("unit/fold must be present and unit must be unique")
            score = float(row["score"])
            if not math.isfinite(score):
                raise ValueError("scores must be finite")
            result[unit] = (fold, score)
    if len(result) < 2:
        raise ValueError("need at least two independent units")
    return result


def quantile(ordered, fraction):
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def compare(baseline, candidate, direction, minimum_gain, resamples, seed, required_fraction):
    if set(baseline) != set(candidate):
        raise ValueError("unit sets differ; intersection-only comparisons are not accepted")
    if not math.isfinite(minimum_gain) or minimum_gain <= 0:
        raise ValueError("minimum gain must be positive and chosen before evaluation")
    if resamples < 100 or not 0 < required_fraction <= 1:
        raise ValueError("need >=100 resamples and fold fraction in (0,1]")
    folds = {}
    gains = []
    sign = 1 if direction == "higher" else -1
    for unit in sorted(baseline):
        fold, before = baseline[unit]
        other_fold, after = candidate[unit]
        if fold != other_fold:
            raise ValueError(f"fold mismatch for unit {unit}")
        delta = sign * (after - before)
        if not math.isfinite(delta):
            raise ValueError("gain overflow")
        gains.append(delta)
        folds.setdefault(fold, []).append(delta)
    if len(folds) < 2:
        raise ValueError("need at least two evaluation folds")
    rng = random.Random(seed)
    samples = sorted(statistics.fmean(rng.choices(gains, k=len(gains))) for _ in range(resamples))
    low, high = quantile(samples, .025), quantile(samples, .975)
    by_fold = {key: statistics.fmean(values) for key, values in folds.items()}
    gain = statistics.fmean(gains)
    failures = []
    if gain < minimum_gain:
        failures.append("gain below predeclared practical minimum")
    if low <= 0:
        failures.append("95% interval includes zero or harm")
    if sum(value > 0 for value in by_fold.values()) / len(folds) < required_fraction:
        failures.append("insufficient agreement across folds")
    return {
        "statistical_screen": "BLOCK" if failures else "PASS", "failures": failures,
        "mean_unit_gain": gain, "interval_95": [low, high], "fold_gains": by_fold,
        "unit_count": len(gains), "worst_unit_gain": min(gains),
        "minimum_gain": minimum_gain, "resamples": resamples, "seed": seed,
        "scope": "equal-weight independent-unit means; no multiple-search correction; not a promotion authorization",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--direction", required=True, choices=["higher", "lower"])
    parser.add_argument("--minimum-gain", required=True, type=float)
    parser.add_argument("--resamples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--fold-fraction", type=float, default=1.0,
                        help="predeclared fraction of folds that must improve")
    args = parser.parse_args()
    try:
        result = compare(read_scores(args.baseline), read_scores(args.candidate), args.direction,
                         args.minimum_gain, args.resamples, args.seed, args.fold_fraction)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["statistical_screen"] == "PASS" else 1)


if __name__ == "__main__":
    main()
