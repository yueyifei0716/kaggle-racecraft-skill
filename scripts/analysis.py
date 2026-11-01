#!/usr/bin/env python3
"""Local, read-only analysis. Every command prints one JSON value; no ledger writes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from metrics import DIRECTIONS, finite, load_scorer, score

STAGES = ["data", "features", "training", "prediction", "scoring"]


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    def reject(value):
        raise ValueError(f"non-finite JSON value: {value}")
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=unique_object, parse_constant=reject)
    def check(item):
        if isinstance(item,float) and not math.isfinite(item):
            raise ValueError("non-finite JSON number")
        if isinstance(item,dict):
            for v in item.values(): check(v)
        elif isinstance(item,list):
            for v in item: check(v)
    check(value)
    return value


def csv_records(path, required):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames
        if not headers or len(headers) != len(set(headers)) or not set(required) <= set(headers):
            raise ValueError(f"{path}: unique headers required, including {','.join(required)}")
        result = []
        for number, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"{path}:{number}: malformed CSV row")
            if any(not row[k].strip() for k in required):
                raise ValueError(f"{path}:{number}: required field is empty")
            result.append(row)
        if not result:
            raise ValueError(f"{path}: no observations")
        return result


def read_units(path):
    rows, seen = csv_records(path, ["unit", "score"]), set()
    for row in rows:
        if row["unit"] in seen:
            raise ValueError(f"duplicate unit: {row['unit']}")
        seen.add(row["unit"])
        row["score"] = finite(row["score"], "score")
        row["weight"] = finite(row.get("weight", "1"), "weight")
        if row["weight"] <= 0:
            raise ValueError("unit weights must be positive")
        for key in ("fold", "group"):
            if key in row and not row[key].strip():
                raise ValueError(f"empty {key}")
    return rows


def read_rows(path):
    rows = csv_records(path, ["row_id", "unit", "fold", "y_true", "y_pred"])
    seen, units = set(), {}
    for row in rows:
        if row["row_id"] in seen:
            raise ValueError(f"duplicate row_id: {row['row_id']}")
        seen.add(row["row_id"])
        row["y_true"] = finite(row["y_true"], "y_true")
        row["y_pred"] = finite(row["y_pred"], "y_pred")
        row["weight"] = finite(row.get("weight", "1"), "weight")
        if row["weight"] <= 0:
            raise ValueError("row weights must be positive")
        identity = (row["fold"], row.get("group"))
        if row["unit"] in units and units[row["unit"]] != identity:
            raise ValueError("one unit cannot span folds or groups")
        units[row["unit"]] = identity
    return rows


def paired(left, right, key):
    a, b = {r[key]: r for r in left}, {r[key]: r for r in right}
    if set(a) != set(b):
        raise ValueError(f"{key} sets do not align exactly")
    result = []
    for name in sorted(a):
        x, y = a[name], b[name]
        for field in ("unit", "fold", "group", "weight", "y_true"):
            if x.get(field) != y.get(field):
                raise ValueError(f"{key}={name}: {field} mismatch")
        result.append((x, y))
    return result


def row_score(rows, metric, custom=None):
    return score(metric, [r["y_true"] for r in rows], [r["y_pred"] for r in rows], [r["weight"] for r in rows], custom)


def weighted_mean(values, weights=None):
    if not values:
        raise ValueError("cannot average an empty sample")
    if weights is None:
        return statistics.mean(values)
    return sum(v*w for v,w in zip(values, weights)) / sum(weights)


def quantile(values, q):
    if not values or not 0 <= q <= 1:
        raise ValueError("quantile requires observations and q in [0,1]")
    ordered = sorted(values)
    position = (len(ordered)-1) * q
    lower = int(position)
    return ordered[lower] + (ordered[min(lower+1, len(ordered)-1)]-ordered[lower]) * (position-lower)


def interval(values, confidence):
    alpha = (1-confidence)/2
    return [quantile(values, alpha), quantile(values, 1-alpha)]


def bootstrap_count(args):
    if args.resamples < 100:
        raise ValueError("at least 100 resamples are required")
    if not 0 < args.confidence < 1:
        raise ValueError("confidence must be in (0,1)")


def ledger(path):
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("ledger must be a JSON object")
    for key in ("metric", "validator", "findings", "candidates", "trust", "holdout"):
        if key not in value:
            raise ValueError(f"ledger missing {key}")
    metric = value["metric"]
    if not isinstance(metric, dict) or metric.get("direction") not in ("higher", "lower") or not isinstance(metric.get("name"), str):
        raise ValueError("metric requires name and direction higher/lower")
    if metric["name"] in DIRECTIONS and metric["direction"] != DIRECTIONS[metric["name"]]:
        raise ValueError("metric direction contradicts built-in definition")
    validator = value["validator"]
    if not isinstance(validator, dict) or validator.get("authority") not in ("full", "veto_only") or not isinstance(validator.get("covered_stages"), list) or any(not isinstance(s, str) for s in validator["covered_stages"]):
        raise ValueError("validator requires authority and covered_stages")
    if not isinstance(value["findings"], list) or not isinstance(value["candidates"], list):
        raise ValueError("findings and candidates must be lists")
    finding_ids = set()
    for item in value["findings"]:
        if not isinstance(item, dict) or any(not isinstance(item.get(k), str) or not item[k] for k in ("id", "severity", "status")):
            raise ValueError("each finding requires id, severity, status")
        if item["id"] in finding_ids:
            raise ValueError("duplicate finding id")
        finding_ids.add(item["id"])
    ids = set()
    for candidate in value["candidates"]:
        if not isinstance(candidate, dict) or any(not isinstance(candidate.get(k), str) or not candidate[k] for k in ("id", "family", "root")):
            raise ValueError("candidate requires nonempty id, family, root")
        if candidate["id"] in ids:
            raise ValueError("duplicate candidate id")
        ids.add(candidate["id"])
        if "structural" in candidate and not isinstance(candidate["structural"], bool):
            raise ValueError("structural must be boolean")
        for key in ("oof", "holdout", "public", "private"):
            if candidate.get(key) is not None:
                if not isinstance(candidate[key],(int,float)) or isinstance(candidate[key],bool):
                    raise ValueError(f"candidate {key} must be a JSON number")
                candidate[key] = finite(candidate[key], key)
        units = candidate.get("selection_units", {})
        if not isinstance(units, dict) or any(not isinstance(k, str) or not k for k in units):
            raise ValueError("selection_units must map nonempty unit names to scores")
        if any(not isinstance(v,(int,float)) or isinstance(v,bool) for v in units.values()):
            raise ValueError("selection unit scores must be JSON numbers")
        candidate["selection_units"] = {k: finite(v, "selection unit score") for k,v in units.items()}
    for c in value["candidates"]:
        if c.get("parent") is not None and c["parent"] not in ids:
            raise ValueError(f"unknown candidate parent: {c['parent']}")
    if value["trust"] is None:
        value["trust"] = {}
    if not isinstance(value["trust"], dict) or not isinstance(value["holdout"], dict):
        raise ValueError("trust and holdout must be objects")
    if value["holdout"].get("opened_at") is not None:
        datetime.fromisoformat(value["holdout"]["opened_at"].replace("Z", "+00:00"))
    return value


def validator_reasons(data, stages):
    reasons = []
    if data["validator"]["authority"] != "full":
        reasons.append("validator_veto_only")
    missing = sorted(set(stages) - set(data["validator"]["covered_stages"]))
    if missing:
        reasons.append("validator_missing_stages:" + ",".join(missing))
    return reasons


def unresolved(data, severities):
    return [f["id"] for f in data["findings"] if f["severity"] in severities and f["status"] != "resolved"]


def scorer(args):
    return load_scorer(args.scorer) if args.scorer else None


def command_score(args):
    rows, custom = read_rows(args.rows), scorer(args)
    unit_metric = args.unit_metric or args.metric
    groups = defaultdict(list)
    for row in rows:
        groups[row["unit"]].append(row)
    units = [{"unit": key, "fold": batch[0]["fold"], "group": batch[0].get("group"), "score": row_score(batch,unit_metric,custom), "rows": len(batch)} for key,batch in sorted(groups.items())]
    return {"metric": args.metric, "unit_metric": unit_metric, "overall": row_score(rows,args.metric,custom), "units": units, "rows": len(rows)}


def matching_rows(path, units, metric, custom):
    rows = read_rows(path)
    by_unit = defaultdict(list)
    for row in rows:
        by_unit[row["unit"]].append(row)
    if set(by_unit) != {u["unit"] for u in units}:
        raise ValueError("row units and unit score file do not align exactly")
    for unit in units:
        batch = by_unit[unit["unit"]]
        for field in ("fold", "group"):
            if unit.get(field) != batch[0].get(field):
                raise ValueError(f"row/unit {field} mismatch")
        if not math.isclose(row_score(batch,metric,custom), unit["score"], rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"row/unit score mismatch: {unit['unit']}")
    return rows


def command_gate(args):
    bootstrap_count(args)
    data = ledger(args.ledger)
    metric, direction = data["metric"]["name"], data["metric"]["direction"]
    unit_metric = args.unit_metric or metric
    unit_direction = DIRECTIONS[unit_metric] if args.unit_metric else direction
    unit_min_gain = args.unit_min_gain if args.unit_metric else max(args.unit_min_gain, args.min_gain)
    unit_ci_min_gain = args.unit_ci_min_gain if args.unit_metric else max(args.unit_ci_min_gain, args.ci_min_gain)
    custom = scorer(args)
    artifact_hash = hashlib.sha256(Path(args.artifact).read_bytes()).hexdigest()
    identified = [c for c in data["candidates"] if c["id"] == args.candidate_id] if args.candidate_id else [c for c in data["candidates"] if c.get("artifact_sha256") == artifact_hash]
    if args.candidate_id and not identified:
        raise ValueError("candidate-id is not registered in ledger")
    for item in identified:
        if item.get("artifact_sha256") != artifact_hash:
            raise ValueError("registered candidate artifact does not match --artifact")
        swept = item.get("swept",1)
        if isinstance(swept,bool):
            swept = 1
        if not isinstance(swept,int) or swept < 1:
            raise ValueError("candidate swept must be an integer >=1")
        args.search_trials = max(args.search_trials,swept)
    left, right = read_units(args.baseline), read_units(args.candidate)
    pairs = paired(left,right,"unit")
    if args.min_folds < 1 or args.min_groups < 1 or args.search_trials < 1 or args.min_gain < 0 or args.ci_min_gain < 0 or args.unit_min_gain < 0 or args.unit_ci_min_gain < 0 or args.max_worst_harm < 0 or args.max_tail_harm < 0 or not 0 <= args.tail_quantile <= 1:
        raise ValueError("gate limits must be nonnegative, counts positive, and tail quantile in [0,1]")
    if bool(args.baseline_rows) != bool(args.candidate_rows):
        raise ValueError("both baseline and candidate row files are required together")
    row_pairs = None
    if args.baseline_rows:
        row_pairs = paired(matching_rows(args.baseline_rows,left,unit_metric,custom),matching_rows(args.candidate_rows,right,unit_metric,custom),"row_id")
    sign = 1 if direction == "higher" else -1
    unit_sign = 1 if unit_direction == "higher" else -1
    deltas = [unit_sign*(y["score"]-x["score"]) for x,y in pairs]
    weights = [x["weight"] for x,_ in pairs]
    clusters = defaultdict(list)
    for i, (x, _) in enumerate(pairs):
        name = x.get(args.cluster)
        if name is None:
            raise ValueError(f"bootstrap cluster {args.cluster} is missing")
        clusters[name].append(i)
    keys, rng = sorted(clusters), random.Random(args.seed)
    row_by_unit = defaultdict(list)
    if row_pairs:
        for pair in row_pairs:
            row_by_unit[pair[0]["unit"]].append(pair)
    samples, overall_samples, invalid = [], [], 0
    for _ in range(args.resamples):
        indices = [i for key in rng.choices(keys,k=len(keys)) for i in clusters[key]]
        samples.append(weighted_mean([deltas[i] for i in indices],[weights[i] for i in indices]))
        if row_pairs:
            rows = [p for i in indices for p in row_by_unit[pairs[i][0]["unit"]]]
            try:
                overall_samples.append(sign*(row_score([p[1] for p in rows],metric,custom)-row_score([p[0] for p in rows],metric,custom)))
            except ValueError:
                invalid += 1
    gain = weighted_mean(deltas, weights)
    ci = interval(samples,args.confidence)
    reasons = validator_reasons(data,args.required_stages)
    blockers = unresolved(data,args.blocking_severities)
    if blockers:
        reasons.append("unresolved_blockers:" + ",".join(blockers))
    folds = {x.get("fold") for x,_ in pairs} - {None}
    groups = {x.get("group") for x,_ in pairs} - {None}
    if len(folds) < args.min_folds:
        reasons.append("insufficient_folds")
    if len(groups) < args.min_groups:
        reasons.append("insufficient_groups")
    if len(keys) < 2:
        reasons.append("insufficient_independent_clusters")
    if gain < unit_min_gain:
        reasons.append("unit_gain_below_minimum")
    if ci[0] <= unit_ci_min_gain:
        reasons.append("unit_ci_not_above_minimum")
    tail = quantile(deltas,args.tail_quantile)
    worst = min(deltas)
    if worst < -args.max_worst_harm:
        reasons.append("worst_unit_harm")
    if tail < -args.max_tail_harm:
        reasons.append("tail_unit_harm")
    replication = None
    if args.search_trials > 1:
        replication = read_json(args.replication) if args.replication else None
        expected = hashlib.sha256(Path(args.artifact).read_bytes()).hexdigest()
        valid = isinstance(replication,dict) and replication.get("fresh") is True and replication.get("independent") is True and replication.get("candidate_sha256") == expected and isinstance(replication.get("evidence_id"),str) and bool(replication["evidence_id"])
        if not valid:
            reasons.append("fresh_independent_replication_required")
    overall = {"status":"unavailable_without_row_predictions"}
    if not row_pairs:
        reasons.append("overall_rows_required")
    if row_pairs:
        overall_gain = sign*(row_score([p[1] for p in row_pairs],metric,custom)-row_score([p[0] for p in row_pairs],metric,custom))
        overall_ci = interval(overall_samples,args.confidence) if overall_samples else None
        overall = {"gain":overall_gain,"ci":overall_ci,"invalid_resamples":invalid,"status":"computed"}
        if invalid:
            reasons.append("undefined_overall_bootstrap_resamples")
        if overall_gain < args.min_gain:
            reasons.append("overall_gain_below_minimum")
        if overall_ci is None or overall_ci[0] <= args.ci_min_gain:
            reasons.append("overall_ci_not_above_minimum")
    return {"decision":"PASS" if not reasons else "BLOCK", "validator_authority":data["validator"]["authority"],"covered_stages":data["validator"]["covered_stages"],"artifact_sha256":hashlib.sha256(Path(args.artifact).read_bytes()).hexdigest(), "evidence":{"reasons":reasons,"metric":data["metric"],"primary_metric":data["metric"],"unit_metric":{"name":unit_metric,"direction":unit_direction},"unit_min_gain":unit_min_gain,"unit_ci_min_gain":unit_ci_min_gain,"overall_min_gain":args.min_gain,"overall_ci_min_gain":args.ci_min_gain,"unit_gain":gain,"unit_ci":ci,"worst_unit_gain":worst,"tail_unit_gain":tail,"folds":len(folds),"groups":len(groups),"clusters":len(keys),"cluster":args.cluster,"overall":overall,"baseline_units_sha256":hashlib.sha256(Path(args.baseline).read_bytes()).hexdigest(),"candidate_units_sha256":hashlib.sha256(Path(args.candidate).read_bytes()).hexdigest(),"baseline_rows_sha256":hashlib.sha256(Path(args.baseline_rows).read_bytes()).hexdigest() if args.baseline_rows else None,"candidate_rows_sha256":hashlib.sha256(Path(args.candidate_rows).read_bytes()).hexdigest() if args.candidate_rows else None,"artifact_binding":"caller supplies artifact; full validator must verify prediction-to-artifact conversion","search_trials":args.search_trials,"replication":replication,"seed":args.seed,"resamples":args.resamples,"confidence":args.confidence}}


def absolute_bootstrap(units, cluster, resamples, confidence, seed, target=None):
    clusters = defaultdict(list)
    for unit in units:
        if unit.get(cluster) is None:
            raise ValueError(f"missing cluster: {cluster}")
        clusters[unit[cluster]].append(unit)
    keys, rng = sorted(clusters), random.Random(seed)
    if len(keys) < 2:
        return {"status":"insufficient_independent_clusters","clusters":len(keys),"mean":weighted_mean([u["score"] for u in units],[u["weight"] for u in units]),"ci":None,"noise_floor":None}
    samples = []
    for _ in range(resamples):
        batch = [u for key in rng.choices(keys,k=target or len(keys)) for u in clusters[key]]
        samples.append(weighted_mean([u["score"] for u in batch],[u["weight"] for u in batch]))
    ci = interval(samples,confidence)
    return {"status":"estimated","clusters":len(keys),"mean":weighted_mean([u["score"] for u in units],[u["weight"] for u in units]),"ci":ci,"noise_floor":(ci[1]-ci[0])/2,"bootstrap_sd":statistics.stdev(samples),"target_independent_clusters":target or len(keys),"assumption":"exchangeable independent clusters from the observed empirical distribution"}


def binary_ordinal_prior(args):
    n = args.target_units or args.simulated_n
    if n is None or n < 2 or args.rows_per_unit < 2 or not 0 < args.prevalence < 1 or not 0 <= args.correlation < 1 or args.effect < 0 or args.classes < 2:
        raise ValueError("prior requires n>=2, rows/unit>=2, prevalence in (0,1), correlation in [0,1), effect>=0 and classes>=2")
    rng, values, invalid = random.Random(args.seed), [],0
    for _ in range(args.resamples):
        truth,pred = [],[]
        for _ in range(n):
            shared=rng.gauss(0,1)*math.sqrt(args.correlation)
            for _ in range(args.rows_per_unit):
                noise=shared+rng.gauss(0,1)*math.sqrt(1-args.correlation)
                if args.prior_metric == "qwk":
                    label=rng.randrange(args.classes)
                    latent=label*args.effect+noise
                    prediction=sum(latent > (k+.5)*args.effect for k in range(args.classes-1))
                else:
                    label=int(rng.random()<args.prevalence)
                    latent=args.effect*(2*label-1)+noise
                    probability=1/(1+math.exp(-max(-700,min(700,latent))))
                    prediction=int(probability>=.5) if args.prior_metric in ("accuracy","f1") else probability
                truth.append(label);pred.append(prediction)
        try:
            values.append(score(args.prior_metric,truth,pred))
        except ValueError:
            invalid+=1
    ci=interval(values,args.confidence) if values else None
    return {"source":"SIMULATED_ASSUMPTIONS_NOT_OBSERVED_EVIDENCE","model":"ordinal_noisy_latent_classes" if args.prior_metric=="qwk" else "binary_gaussian_margin_logistic_probability", "metric":args.prior_metric,"assumptions":{"independent_units":n,"rows_per_unit":args.rows_per_unit,"prevalence":args.prevalence if args.prior_metric!="qwk" else None,"classes":args.classes if args.prior_metric=="qwk" else 2,"effect":args.effect,"within_unit_latent_noise_correlation":args.correlation,"class_distribution":"uniform" if args.prior_metric=="qwk" else "bernoulli","labels_independent_given_unit_noise":True},"status":"simulated" if values and not invalid else "undefined_simulation_draws","mean":statistics.mean(values) if values else None,"score_ci":ci,"noise_floor":(ci[1]-ci[0])/2 if ci else None,"valid_draws":len(values),"invalid_draws":invalid,"seed":args.seed,"resamples":args.resamples,"limitation":"Synthetic model assumptions are not estimates of competition prevalence, calibration, dependence or test scores."}


def command_noise(args):
    bootstrap_count(args)
    if args.target_units is not None and args.target_units < 2:
        raise ValueError("target units must be at least 2 independent clusters")
    if args.units:
        units = read_units(args.units)
        result = absolute_bootstrap(units,args.cluster,args.resamples,args.confidence,args.seed,args.target_units)
        result.update({"source":"observed_absolute_unit_scores","cluster":args.cluster,"seed":args.seed,"confidence":args.confidence})
    else:
        if args.simulated_sd is None and args.prior_metric is None:
            raise ValueError("noise requires --units, --simulated-sd, or an explicit --prior-metric model")
        result = {"source":"simulation_only","status":"no_observed_evidence"}
    if args.simulated_sd is not None:
        n = args.simulated_n or args.target_units
        if args.simulated_sd < 0 or n is None or n < 2:
            raise ValueError("simulation requires nonnegative sd and n >= 2")
        rng = random.Random(args.seed)
        draws = [rng.gauss(0,args.simulated_sd/math.sqrt(n)) for _ in range(args.resamples)]
        ci = interval(draws,args.confidence)
        result["simulation_prior"] = {"source":"SIMULATED_PRIOR_NOT_OBSERVED_EVIDENCE","distribution":"normal_iid_mean","assumed_sd":args.simulated_sd,"assumed_n":n,"centered_ci":ci,"noise_floor":(ci[1]-ci[0])/2,"limitation":"Parameterized score-scale SD prior; no binary/ordinal metric-specific model or inferred test distribution."}
    elif args.simulated_n is not None and args.prior_metric is None:
        raise ValueError("simulated-n requires simulated-sd or prior-metric")
    if args.prior_metric:
        result["metric_simulation_prior"] = binary_ordinal_prior(args)
    return result


def command_groups(args):
    bootstrap_count(args)
    files = {}
    if args.units:
        files["observed"] = args.units
    for spec in args.candidate or []:
        name,sep,path = spec.partition("=")
        if not sep or not name or not path or name in files:
            raise ValueError("group candidates must be unique NAME=CSV pairs")
        files[name] = path
    if not files:
        raise ValueError("groups requires --units or --candidate NAME=CSV")
    candidates = {name:read_units(path) for name,path in files.items()}
    first = next(iter(candidates.values()))
    for batch in candidates.values():
        paired(first,batch,"unit")
    overall, grouped = {},{}
    for name,units in candidates.items():
        groups = defaultdict(list)
        for unit in units:
            if "group" not in unit:
                raise ValueError("group report requires a group for every unit")
            groups[unit["group"]].append(unit)
        grouped[name] = {key:absolute_bootstrap(batch,"unit",args.resamples,args.confidence,args.seed) for key,batch in sorted(groups.items())}
        overall[name] = absolute_bootstrap(units,"group",args.resamples,args.confidence,args.seed)
    sign = 1 if args.direction == "higher" else -1
    group_names = sorted(next(iter(grouped.values())))
    rankings, flips = {},[]
    names = list(candidates)
    for group in group_names:
        ranked = sorted(names,key=lambda n:(-sign*grouped[n][group]["mean"],n))
        rankings[group] = [{"candidate":n,"unit_mean_score":grouped[n][group]["mean"],"rank":1+sum(sign*grouped[o][group]["mean"] > sign*grouped[n][group]["mean"] for o in names)} for n in ranked]
    for i,a in enumerate(names):
        for b in names[i+1:]:
            overall_gain = sign*(overall[b]["mean"]-overall[a]["mean"])
            for group in group_names:
                group_gain = sign*(grouped[b][group]["mean"]-grouped[a][group]["mean"])
                if overall_gain*group_gain < 0:
                    flips.append({"baseline":a,"candidate":b,"group":group,"overall_unit_mean_gain":overall_gain,"group_unit_mean_gain":group_gain})
    result = {"candidates":{n:{"overall":overall[n],"groups":grouped[n]} for n in names},"group_rankings":rankings,"sign_flips":flips,"direction":args.direction,"interpretation":"weighted means of unit scores, not pooled nonadditive metrics; descriptive group comparisons, no multiple-testing correction"}
    if len(names)==1:
        n=names[0]
        result.update({"groups":[dict(group=k,units=sum(u["group"]==k for u in first),**v) for k,v in grouped[n].items()],"overall":overall[n]})
    return result


def command_null(args):
    bootstrap_count(args)
    rows, custom = read_rows(args.rows), scorer(args)
    strata = defaultdict(list)
    for i,row in enumerate(rows):
        strata[(row["unit"],row["fold"])].append(i)
    if all(len(v) < 2 or len({rows[i]["y_true"] for i in v}) < 2 or len({rows[i]["y_pred"] for i in v}) < 2 for v in strata.values()):
        return {"status":"insufficient_exchangeable_variation","strata":len(strata),"null_ci":None,"p_value":None}
    rng, nulls = random.Random(args.seed), []
    for _ in range(args.resamples):
        copied = [dict(r) for r in rows]
        for indices in strata.values():
            predictions = [rows[i]["y_pred"] for i in indices]
            rng.shuffle(predictions)
            for i,p in zip(indices,predictions):
                copied[i]["y_pred"] = p
        nulls.append(row_score(copied,args.metric,custom))
    actual = row_score(rows,args.metric,custom)
    direction = args.direction or DIRECTIONS.get(args.metric)
    if direction is None:
        raise ValueError("custom null metric requires --direction")
    exceed = sum(v >= actual if direction == "higher" else v <= actual for v in nulls)
    return {"status":"computed","observed":actual,"null_mean":statistics.mean(nulls),"null_ci":interval(nulls,args.confidence),"p_value":(exceed+1)/(len(nulls)+1),"resamples":len(nulls),"shuffle_within":["unit","fold"],"exchangeability":"assumed; verify dependence before interpreting p_value"}


def ranks(values):
    ordered, result = sorted(range(len(values)),key=lambda i:values[i]), [0.0]*len(values)
    i = 0
    while i < len(values):
        j = i+1
        while j < len(values) and values[ordered[j]] == values[ordered[i]]:
            j += 1
        for index in ordered[i:j]:
            result[index] = (i+j-1)/2+1
        i = j
    return result


def correlation(a,b,min_n=3):
    if len(a) != len(b):
        raise ValueError("correlation arrays do not align")
    if len(a) < min_n:
        return {"status":"insufficient_samples","n":len(a),"rho":None}
    if len(set(a)) < 2 or len(set(b)) < 2:
        return {"status":"constant_scores","n":len(a),"rho":None}
    x,y = ranks(a),ranks(b)
    mx,my = statistics.mean(x),statistics.mean(y)
    numerator = sum((a-mx)*(b-my) for a,b in zip(x,y))
    denominator = math.sqrt(sum((v-mx)**2 for v in x)*sum((v-my)**2 for v in y))
    return {"status":"computed","n":len(a),"rho":numerator/denominator}


def pearson(a,b,min_n=3):
    if len(a) != len(b):
        raise ValueError("residual arrays do not align")
    if len(a) < min_n:
        return {"status":"insufficient_samples","n":len(a),"correlation":None}
    if len(set(a)) < 2 or len(set(b)) < 2:
        return {"status":"constant_residuals","n":len(a),"correlation":None}
    ma,mb = statistics.mean(a),statistics.mean(b)
    denominator = math.sqrt(sum((v-ma)**2 for v in a)*sum((v-mb)**2 for v in b))
    return {"status":"computed","n":len(a),"correlation":sum((x-ma)*(y-mb) for x,y in zip(a,b))/denominator}


def metric_pairs(candidates,left,right):
    return [c for c in candidates if c.get(left) is not None and c.get(right) is not None]


def trust_fingerprint(data,local):
    fields = ["id","family","root","parent","structural",local,"public","artifact_sha256","gate_artifact_sha256"]
    candidates = [{k:c.get(k) for k in fields} for c in metric_pairs(data["candidates"],local,"public")]
    payload = {"metric":data["metric"],"validator":data["validator"],"findings":data["findings"],"holdout":data["holdout"],"candidates":sorted(candidates,key=lambda c:c["id"]),"public_units":data.get("public_units",data["trust"].get("public_units"))}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def public_hedge_reasons(data,authority,args):
    trust = data["trust"]
    reasons = []
    if trust.get("regime") not in ("ALIGNED","WEAK"):
        reasons.append("public_trust_regime_not_eligible")
    if trust.get("local") != authority:
        reasons.append("trust_local_authority_changed")
    if trust.get("candidate_fingerprint") != trust_fingerprint(data,authority):
        reasons.append("trust_fingerprint_stale_or_missing")
    paired_ids = sorted(c["id"] for c in metric_pairs(data["candidates"],authority,"public"))
    if sorted(trust.get("candidate_ids",[])) != paired_ids or trust.get("n_paired") != len(paired_ids) or len(paired_ids) < args.min_trust_samples:
        reasons.append("trust_paired_candidates_insufficient_or_changed")
    n = data.get("public_units",trust.get("public_units"))
    if not isinstance(n,int) or isinstance(n,bool) or n < args.min_public_units:
        reasons.append("public_units_insufficient")
    try:
        computed = datetime.fromisoformat(trust["computed_at"].replace("Z","+00:00"))
        age = (datetime.now(timezone.utc)-computed).total_seconds()/3600
        if not 0 <= age <= args.trust_max_age_hours:
            reasons.append("trust_timestamp_stale")
    except (KeyError,ValueError,TypeError,AttributeError):
        reasons.append("trust_timestamp_missing_or_invalid")
    return reasons


def command_trust(args):
    data = ledger(args.ledger)
    if args.min_samples < 3 or args.min_public_units < 1 or not -1 <= args.min_rho <= 1:
        raise ValueError("trust requires min-samples >= 3 and min-public-units >= 1")
    candidates = metric_pairs(data["candidates"],args.local,"public")
    def measure(batch):
        return correlation([c[args.local] for c in batch],[c["public"] for c in batch],args.min_samples)
    families = defaultdict(list)
    for candidate in candidates:
        families[candidate["family"]].append(candidate)
    overall = measure(candidates)
    structural = measure([c for c in candidates if c.get("structural") is True])
    within = {name:measure(batch) for name,batch in sorted(families.items())}
    public_units = data.get("public_units",data["trust"].get("public_units"))
    limitations = []
    if public_units is None:
        limitations.append("public_units_unknown")
    elif not isinstance(public_units,int) or isinstance(public_units,bool) or public_units < 1:
        raise ValueError("trust.public_units must be a positive integer")
    elif public_units < args.min_public_units:
        limitations.append("public_units_below_minimum")
    if any(m["status"] != "computed" for m in [overall,structural]):
        limitations.append("overall_or_structural_unidentified")
    if any(m["status"] != "computed" for m in within.values()):
        limitations.append("within_family_unidentified")
    computed = [m["rho"] for m in [overall,structural,*within.values()] if m["status"] == "computed"]
    if computed and min(computed) < args.min_rho:
        limitations.append("rank_agreement_below_minimum")
    regime = "INSUFFICIENT" if any(v in limitations for v in ("public_units_unknown","public_units_below_minimum","overall_or_structural_unidentified","within_family_unidentified")) else "DIVERGENT" if computed and min(computed) < 0 else "WEAK" if limitations else "ALIGNED"
    return {"local":args.local,"overall":overall,"structural":structural,"within_family":within,"public_units":public_units,"regime":regime,"candidate_ids":[c["id"] for c in candidates],"computed_at":datetime.now(timezone.utc).isoformat(),"n_paired":len(candidates),"candidate_fingerprint":trust_fingerprint(data,args.local),"reason":";".join(limitations) or "rank_agreement_observed","limitations":limitations,"selection_authority":"local_holdout_or_oof","caution":"Correlation is descriptive; adaptive search and public sampling are not corrected."}


def command_leaderboard(args):
    rows = csv_records(args.table,["id","score"])
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate leaderboard id")
    for row in rows:
        row["score"] = finite(row["score"],"leaderboard score")
    thresholds = read_json(args.thresholds)
    if not isinstance(thresholds,dict) or not thresholds:
        raise ValueError("thresholds must be a nonempty object of user-specified medal:score")
    thresholds = {k:finite(v,"medal threshold") for k,v in thresholds.items()}
    sign = 1 if args.direction == "higher" else -1
    order = sorted(rows,key=lambda r:-sign*r["score"])
    platforms = defaultdict(list)
    for row in order:
        platforms[row["score"]].append(row["id"])
    result, position = [],1
    for value,members in platforms.items():
        result.append({"score":value,"ids":members,"rank_start":position,"rank_end":position+len(members)-1,"medals_meeting_score_threshold":[name for name,limit in thresholds.items() if sign*value >= sign*limit]})
        position += len(members)
    return {"platforms":result,"thresholds":thresholds,"threshold_source":"explicit_user_input","tie_policy":"score-only eligibility; official tie/rank eligibility remains unspecified","official_awards_inferred":False}


def valid_hash(value):
    return isinstance(value,str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def candidate_gate_reasons(candidate,stages,metric):
    reasons = []
    gate = candidate.get("gate")
    if not isinstance(gate,dict) or gate.get("decision") != "PASS":
        return ["promotion_gate_not_passed"]
    if gate.get("validator_authority") != "full" or not set(stages) <= set(gate.get("covered_stages",[])):
        reasons.append("gate_validator_incomplete")
    if not isinstance(gate.get("evidence"),dict) or gate["evidence"].get("metric") != metric:
        reasons.append("gate_metric_stale_or_missing")
    artifact, checked = candidate.get("artifact_sha256"),candidate.get("gate_artifact_sha256")
    if not valid_hash(artifact) or artifact != checked or artifact != gate.get("artifact_sha256"):
        reasons.append("gate_artifact_identity_stale_or_missing")
    swept = candidate.get("swept",1)
    if isinstance(swept,bool):
        swept = 1
    if not isinstance(swept,int) or swept < 1:
        return reasons+["invalid_search_count"]
    if swept > 1:
        evidence = gate.get("evidence",{})
        trials = evidence.get("search_trials")
        receipt = evidence.get("replication")
        if not isinstance(trials,int) or isinstance(trials,bool) or trials < swept or not isinstance(receipt,dict) or receipt.get("fresh") is not True or receipt.get("independent") is not True or receipt.get("candidate_sha256") != artifact or not isinstance(receipt.get("evidence_id"),str) or not receipt["evidence_id"]:
            reasons.append("fresh_replication_or_search_count_missing")
    return reasons


def command_select(args):
    data = ledger(args.ledger)
    if args.count < 1 or not -1 <= args.max_correlation <= 1 or args.min_shared_units < 3 or args.public_max_local_regret < 0 or args.min_trust_samples < 3 or args.min_public_units < 1 or args.trust_max_age_hours <= 0:
        raise ValueError("selection requires count>=1, correlation in [-1,1], shared units>=3")
    global_reasons = validator_reasons(data,args.required_stages)
    blockers = unresolved(data,args.blocking_severities)
    if blockers:
        global_reasons.append("unresolved_blockers:"+",".join(blockers))
    use_holdout = data["holdout"].get("opened_at") is not None
    authority = "holdout" if use_holdout else "oof"
    eligible, excluded = [],{}
    for c in data["candidates"]:
        reasons = candidate_gate_reasons(c,args.required_stages,data["metric"])
        if c.get("status") not in ("validated","eligible","promoted","selected"):
            reasons.append("candidate_not_validated")
        if c.get(authority) is None:
            reasons.append("missing_authoritative_score")
        if c.get("public") is None:
            reasons.append("public_score_not_retrieved")
        if reasons:
            excluded[c["id"]] = reasons
        else:
            eligible.append(c)
    sign = 1 if data["metric"]["direction"] == "higher" else -1
    eligible.sort(key=lambda c:(-sign*c[authority],c["id"]))
    baseline = {u["unit"]:u["score"] for u in read_units(args.residual_baseline)} if args.residual_baseline else {}
    hedge_reasons = public_hedge_reasons(data,authority,args) if args.public_second else ["public_second_not_requested"]
    hedge = args.public_second and not hedge_reasons
    if hedge and eligible:
        best = eligible[0]
        credible = [c for c in eligible[1:] if c.get("public") is not None and sign*(best[authority]-c[authority]) <= args.public_max_local_regret]
        credible.sort(key=lambda c:(-sign*c["public"],-sign*c[authority],c["id"]))
        eligible = [best]+credible+[c for c in eligible[1:] if c not in credible]
    selected, diversity = [],[]
    queue = list(eligible)
    while queue and len(selected) < args.count:
        if len(selected) >= 2:
            queue.sort(key=lambda c:(-sign*c[authority],c["id"]))
        candidate=queue.pop(0)
        checks, reject = [],False
        for chosen in selected:
            if candidate["root"] == chosen["root"]:
                checks.append({"against":chosen["id"],"status":"same_root"})
                reject = True
                continue
            a,b = candidate["selection_units"],chosen["selection_units"]
            common = sorted(set(a)&set(b))
            if not baseline or set(a) != set(b) or set(a) != set(baseline):
                checks.append({"against":chosen["id"],"status":"residual_baseline_or_unit_alignment_missing"})
                reject = True
                continue
            corr = pearson([a[k]-baseline[k] for k in common],[b[k]-baseline[k] for k in common],args.min_shared_units)
            checks.append({"against":chosen["id"],**corr})
            if set(a) != set(b) or corr["status"] != "computed" or corr["correlation"] > args.max_correlation:
                reject = True
        diversity.append({"candidate":candidate["id"],"checks":checks,"accepted":not reject})
        if reject:
            excluded[candidate["id"]] = ["root_or_residual_diversity_not_established"]
        else:
            selected.append(candidate)
    if global_reasons:
        selected = []
    return {"status":"blocked" if global_reasons else "recommendation_only","reasons":global_reasons,"authority":authority,"public_role":"diverse_second_with_local_regret_limit" if hedge else "diagnostic_only","public_hedge_reasons":hedge_reasons,"recommendations":[{"id":c["id"],"score":c[authority],"root":c["root"]} for c in selected],"candidate_ids":[c["id"] for c in selected],"requested":args.count,"unfilled":max(0,args.count-len(selected)),"excluded":excluded,"diversity":diversity,"execution":"none"}


def command_postmortem(args):
    if args.min_samples < 3:
        raise ValueError("postmortem requires min-samples >= 3")
    data = ledger(args.ledger)
    candidates = data["candidates"]
    paired_candidates = metric_pairs(candidates,"public","private")
    correlation_result = correlation([c["public"] for c in paired_candidates],[c["private"] for c in paired_candidates],args.min_samples)
    evaluated = [c for c in candidates if c.get("private") is not None]
    sign = 1 if data["metric"]["direction"] == "higher" else -1
    evaluated.sort(key=lambda c:(-sign*c["private"],c["id"]))
    selected = set(args.selected or [c["id"] for c in candidates if c.get("status") == "selected"])
    if selected-set(c["id"] for c in candidates):
        raise ValueError("unknown selected candidate id")
    selected_evaluated = [c for c in evaluated if c["id"] in selected]
    unselected = [c for c in evaluated if c["id"] not in selected]
    best = unselected[0] if unselected else None
    best_selected = selected_evaluated[0] if selected_evaluated else None
    frontier = evaluated[0] if evaluated else None
    decomposition = {"target_score":args.target_score,"best_evaluated_id":frontier["id"] if frontier else None,"model_frontier_gap":sign*(args.target_score-frontier["private"]) if frontier and args.target_score is not None else None,"selection_regret":sign*(frontier["private"]-best_selected["private"]) if frontier and best_selected else None,"total_gap":sign*(args.target_score-best_selected["private"]) if best_selected and args.target_score is not None else None,"interpretation":"signed total_gap = model_frontier_gap + selection_regret; evaluated candidates only; descriptive structural/adaptive labels do not establish causes"}
    return {"public_private_spearman":correlation_result,"best_unselected":None if best is None else {"id":best["id"],"private":best["private"]},"best_selected":None if best_selected is None else {"id":best_selected["id"],"private":best_selected["private"]},"missed_gain":None if best is None or best_selected is None else sign*(best["private"]-best_selected["private"]),"gap_decomposition":decomposition,"private_missing":[c["id"] for c in candidates if c.get("private") is None],"interpretation":"retrospective descriptive evidence; does not change a historical selection"}


def comma_list(value):
    values = value.split(",")
    if any(not v for v in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of unique nonempty names")
    return values


def add_bootstrap(parser):
    parser.add_argument("--resamples",type=int,default=2000)
    parser.add_argument("--confidence",type=finite,default=.95)
    parser.add_argument("--seed",type=int,default=42)


def add_metric(parser):
    parser.add_argument("--metric",choices=[*DIRECTIONS,"custom"],required=True)
    parser.add_argument("--scorer",help="trusted local scorer.py:function; called with truth,predictions,weights")


def add_policy(parser):
    parser.add_argument("--required-stages",type=comma_list,default=STAGES)
    parser.add_argument("--blocking-severities",type=comma_list,default=["blocker","critical","high"])


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command",required=True)
    p = sub.add_parser("score",help="score aligned row predictions overall and per unit")
    p.add_argument("--rows",required=True); add_metric(p)
    p.add_argument("--unit-metric",choices=list(DIRECTIONS),help="explicit per-unit metric; pooled overall keeps --metric")
    p.set_defaults(run=command_score)
    p = sub.add_parser("gate",help="paired unit/pooled bootstrap promotion recommendation")
    p.add_argument("--ledger",required=True)
    p.add_argument("--baseline",required=True); p.add_argument("--candidate",required=True)
    p.add_argument("--artifact",required=True,help="actual frozen candidate prediction/submission artifact to bind by SHA256")
    p.add_argument("--candidate-id",help="registered candidate; declared swept count cannot be reduced")
    p.add_argument("--baseline-rows"); p.add_argument("--candidate-rows")
    p.add_argument("--scorer")
    p.add_argument("--unit-metric",choices=list(DIRECTIONS),help="explicit unit score metric; overall keeps the ledger metric")
    p.add_argument("--cluster",choices=["unit","group","fold"],default="unit")
    p.add_argument("--min-gain",type=finite,default=0)
    p.add_argument("--ci-min-gain",type=finite,default=0)
    p.add_argument("--unit-min-gain",type=finite,default=0)
    p.add_argument("--unit-ci-min-gain",type=finite,default=0)
    p.add_argument("--min-folds",type=int,default=3); p.add_argument("--min-groups",type=int,default=2)
    p.add_argument("--max-worst-harm",type=finite,default=0); p.add_argument("--max-tail-harm",type=finite,default=0)
    p.add_argument("--tail-quantile",type=finite,default=.1)
    p.add_argument("--search-trials",type=int,default=1)
    p.add_argument("--replication",help="JSON fresh/independent/evidence_id/candidate_sha256 receipt; provenance remains user responsibility")
    add_bootstrap(p); add_policy(p); p.set_defaults(run=command_gate)
    p = sub.add_parser("noise",help="absolute-score independent cluster uncertainty and labeled simulation prior")
    p.add_argument("--units"); p.add_argument("--cluster",choices=["unit","group","fold"],default="unit")
    p.add_argument("--target-units",type=int,help="target number of independent clusters (groups when --cluster group)")
    p.add_argument("--simulated-sd",type=finite); p.add_argument("--simulated-n",type=int)
    p.add_argument("--prior-metric",choices=["accuracy","f1","logloss","auc","qwk"])
    p.add_argument("--prevalence",type=finite,default=.5); p.add_argument("--classes",type=int,default=3)
    p.add_argument("--effect",type=finite,default=1); p.add_argument("--rows-per-unit",type=int,default=32)
    p.add_argument("--correlation",type=finite,default=0,help="within-unit Gaussian latent noise correlation, not empirical label correlation")
    add_bootstrap(p); p.set_defaults(run=command_noise)
    p = sub.add_parser("groups",help="absolute-score group report")
    p.add_argument("--units"); p.add_argument("--candidate",action="append",help="NAME=unit_scores.csv, repeat for comparisons")
    p.add_argument("--direction",choices=["higher","lower"],default="lower")
    add_bootstrap(p); p.set_defaults(run=command_groups)
    p = sub.add_parser("null",help="shuffle predictions within each unit/fold, assuming exchangeability")
    p.add_argument("--rows",required=True); add_metric(p); add_bootstrap(p)
    p.add_argument("--direction",choices=["higher","lower"]); p.set_defaults(run=command_null)
    p = sub.add_parser("trust",help="overall/structural/within-family public-local Spearman")
    p.add_argument("--ledger",required=True); p.add_argument("--local",choices=["oof","holdout"],default="oof")
    p.add_argument("--min-samples",type=int,default=4); p.add_argument("--min-public-units",type=int,default=20)
    p.add_argument("--min-rho",type=finite,default=.5); p.set_defaults(run=command_trust)
    p = sub.add_parser("leaderboard",help="user-specified medal score thresholds and tied score platforms")
    p.add_argument("--table",required=True); p.add_argument("--thresholds",required=True)
    p.add_argument("--direction",choices=["higher","lower"],required=True); p.set_defaults(run=command_leaderboard)
    p = sub.add_parser("select",help="recommend final candidates using holdout/OOF, blockers and diversity")
    p.add_argument("--ledger",required=True); p.add_argument("--count",type=int,default=2)
    p.add_argument("--residual-baseline",help="unit score CSV used to remove common unit difficulty for residual correlation")
    p.add_argument("--public-second",action="store_true",help="allow fresh ALIGNED/WEAK public evidence to rank a credible diverse second slot")
    p.add_argument("--public-max-local-regret",type=finite,default=0,help="maximum local score regret allowed for public second")
    p.add_argument("--min-trust-samples",type=int,default=4); p.add_argument("--min-public-units",type=int,default=20)
    p.add_argument("--trust-max-age-hours",type=finite,default=24)
    p.add_argument("--max-correlation",type=finite,default=.95); p.add_argument("--min-shared-units",type=int,default=4)
    add_policy(p); p.set_defaults(run=command_select)
    p = sub.add_parser("postmortem",help="retrospective public/private rank correlation and best unselected")
    p.add_argument("--ledger",required=True); p.add_argument("--selected",type=comma_list)
    p.add_argument("--min-samples",type=int,default=4); p.set_defaults(run=command_postmortem)
    p.add_argument("--target-score",type=finite,help="explicit retrospective target/frontier for signed gap decomposition")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = args.run(args)
        print(json.dumps(result,ensure_ascii=False,sort_keys=True,allow_nan=False))
        return 0
    except (ValueError,KeyError,TypeError,OSError,AttributeError) as exc:
        print(json.dumps({"error":str(exc)},ensure_ascii=False),file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
