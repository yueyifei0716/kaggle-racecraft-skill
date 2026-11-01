"""Synthetic fixtures only: no credentials, network, datasets or ledger mutations."""
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0,str(SCRIPTS))
import analysis
from metrics import score


class MetricTests(unittest.TestCase):
    def test_mae_and_rmse_weights(self):
        self.assertEqual(score("mae",[0,0],[1,3],[3,1]),1.5)
        self.assertAlmostEqual(score("rmse",[0,0],[1,3],[3,1]),math.sqrt(3))

    def test_accuracy(self):
        self.assertEqual(score("accuracy",[0,1,2],[0,0,2],[1,2,1]),.5)

    def test_binary_f1_weighted_and_zero(self):
        self.assertAlmostEqual(score("f1",[1,1,0],[1,0,1],[2,1,1]),2/3)
        self.assertEqual(score("f1",[0,0],[0,0]),0)
        with self.assertRaises(ValueError):
            score("f1",[0,1],[.2,.8])

    def test_binary_logloss(self):
        self.assertAlmostEqual(score("logloss",[0,1],[.25,.75]),-math.log(.75))
        self.assertTrue(math.isfinite(score("logloss",[0,1],[1,0])))

    def test_auc_ties_and_weights(self):
        self.assertEqual(score("auc",[0,1],[.5,.5]),.5)
        self.assertAlmostEqual(score("auc",[0,0,1,1],[.1,.8,.2,.9],[1,2,3,4]),15/21)
        self.assertEqual(score("auc",[0,1],[0,1]),1)

    def test_qwk(self):
        self.assertEqual(score("qwk",[0,1,2],[0,1,2]),1)
        self.assertAlmostEqual(score("qwk",[0,1,2],[2,1,0]),-1)
        self.assertAlmostEqual(score("qwk",[0,2],[0,0]),0)

    def test_invalid_metric_inputs(self):
        for name,y,p,w in [("mae",[],[],None),("mae",[0],[1,2],None),("mae",[0],[math.inf],None),("mae",[0],[0],[0]),("mae",[0],[0],[-1]),("mae",[0],[0],[]),("auc",[1],[.2],None),("auc",[2,1],[.2,.4],None),("logloss",[0],[1.1],None),("qwk",[.1],[.1],None),("qwk",[1],[1],None)]:
            with self.subTest(name=name,y=y,p=p,w=w), self.assertRaises(ValueError):
                score(name,y,p,w)

    def test_custom(self):
        self.assertEqual(score("custom",[1,2],[3,4],custom=lambda y,p,w:sum(p)-sum(y)),4)
        with self.assertRaises(ValueError):
            score("custom",[1],[1],custom=lambda y,p,w:math.nan)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.data = {"metric":{"name":"mae","direction":"lower"},"validator":{"authority":"full","covered_stages":analysis.STAGES},"findings":[],"candidates":[],"trust":{"regime":"INSUFFICIENT","public_units":100},"holdout":{"opened_at":None}}
        self.ledger_path = self.jsonfile("ledger.json",self.data)

    def tearDown(self):
        self.temp.cleanup()

    def jsonfile(self,name,data):
        path = self.path/name
        path.write_text(json.dumps(data),encoding="utf-8")
        return path

    def csvfile(self,name,header,rows):
        path = self.path/name
        with path.open("w",newline="",encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header); writer.writerows(rows)
        return path

    def run_cli(self,*args,ok=True):
        completed = subprocess.run([sys.executable,str(SCRIPTS/"analysis.py"),*map(str,args)],capture_output=True,text=True)
        self.assertEqual(completed.returncode,0 if ok else 2,completed.stderr)
        return json.loads(completed.stdout if ok else completed.stderr)

    def units(self,name="units.csv",scores=None):
        return self.csvfile(name,["unit","score","weight","fold","group"],[[str(i),v,1,str(i%3),str(i%2)] for i,v in enumerate(scores if scores is not None else [1,2,3,4,5,6])])

    def gate_fixture(self,baseline=None,candidate=None):
        baseline = baseline if baseline is not None else [2]*6
        candidate = candidate if candidate is not None else [1]*6
        a,b = self.units("baseline.csv",baseline),self.units("candidate.csv",candidate)
        header = ["row_id","unit","fold","group","y_true","y_pred"]
        ar = self.csvfile("base_rows.csv",header,[[str(i),str(i),str(i%3),str(i%2),0,v] for i,v in enumerate(baseline)])
        br = self.csvfile("cand_rows.csv",header,[[str(i),str(i),str(i%3),str(i%2),0,v] for i,v in enumerate(candidate)])
        return ["gate","--ledger",self.ledger_path,"--baseline",a,"--candidate",b,"--artifact",br,"--baseline-rows",ar,"--candidate-rows",br,"--resamples","100"]

    def candidate(self,name,score_value=1,root=None,units=None):
        digest = hashlib.sha256(name.encode()).hexdigest()
        return {"id":name,"family":root or name,"root":root or name,"parent":None,"structural":True,"oof":score_value,"holdout":None,"public":score_value,"private":None,"status":"validated","gate":{"decision":"PASS","validator_authority":"full","covered_stages":analysis.STAGES,"artifact_sha256":digest,"evidence":{"metric":{"name":"mae","direction":"lower"}}},"artifact_sha256":digest,"gate_artifact_sha256":digest,"selection_units":units or {"a":1,"b":2,"c":3,"d":4}}

    def save(self):
        self.jsonfile("ledger.json",self.data)

    def test_score_cli_overall_and_units(self):
        rows=self.csvfile("rows.csv",["row_id","unit","fold","y_true","y_pred"],[["a","x","0",0,3],["b","x","0",0,4],["c","y","1",0,0]])
        result=self.run_cli("score","--rows",rows,"--metric","rmse")
        self.assertAlmostEqual(result["overall"],math.sqrt(25/3))
        self.assertEqual(result["units"][0]["score"],math.sqrt(12.5))
        self.assertEqual(result["rows"],3)

    def secondary_fixture(self, metric="auc", unit_metric="logloss", regression=False):
        self.data["metric"]={"name":metric,"direction":"higher"}; self.save()
        header=["row_id","unit","fold","group","y_true","y_pred"]
        base_rows,candidate_rows=[],[]
        for index in range(40):
            label=index%2
            if metric == "auc":
                base=(.1 if label == 0 else .09) if regression else (.6 if label == 0 else .4)
                candidate=(.99 if label == 0 else .995) if regression else (.2 if label == 0 else .8)
            else:
                label*=2
                base,candidate=1,label
            identity=[f"r{index}",str(index),str(index%3),f"patient-{index}",label]
            base_rows.append(identity+[base]); candidate_rows.append(identity+[candidate])
        a=self.csvfile("secondary-base-rows.csv",header,base_rows)
        b=self.csvfile("secondary-candidate-rows.csv",header,candidate_rows)
        results=[self.run_cli("score","--rows",path,"--metric",metric,"--unit-metric",unit_metric) for path in (a,b)]
        paths=[]
        for name,result in zip(("secondary-base-units.csv","secondary-candidate-units.csv"),results):
            paths.append(self.csvfile(name,["unit","fold","group","score"],
                [[u["unit"],u["fold"],u["group"],u["score"]] for u in result["units"]]))
        args=["gate","--ledger",self.ledger_path,"--baseline",paths[0],"--candidate",paths[1],
              "--baseline-rows",a,"--candidate-rows",b,"--artifact",b,"--resamples","300", "--unit-metric",unit_metric]
        return args,results,(a,b),paths

    def test_auc_single_label_units_use_explicit_logloss_and_preserve_pooled_auc(self):
        args,results,rows,_=self.secondary_fixture()
        baseline,candidate=results
        self.assertEqual(baseline["metric"],"auc")
        self.assertEqual(candidate["unit_metric"],"logloss")
        self.assertEqual((baseline["overall"],candidate["overall"]),(0,1))
        self.assertEqual(len(candidate["units"]),40)
        self.assertTrue(all(u["rows"] == 1 for u in candidate["units"]))
        self.assertTrue(all(math.isclose(u["score"],-math.log(.8)) for u in candidate["units"]))
        failed=self.run_cli("score","--rows",rows[1],"--metric","auc",ok=False)
        self.assertIn("AUC requires both classes",failed["error"])
        result=self.run_cli(*args)
        self.assertEqual(result["decision"],"PASS")
        evidence=result["evidence"]
        self.assertEqual(evidence["metric"],{"name":"auc","direction":"higher"})
        self.assertEqual(evidence["primary_metric"],evidence["metric"])
        self.assertEqual(evidence["unit_metric"],{"name":"logloss","direction":"lower"})
        self.assertAlmostEqual(evidence["unit_gain"],math.log(2))
        self.assertEqual(evidence["overall"]["gain"],1)
        self.assertEqual(evidence["overall"]["invalid_resamples"],0)

    def test_secondary_regression_blocks_even_when_primary_auc_improves(self):
        args,results,_,_=self.secondary_fixture(regression=True)
        self.assertGreater(results[1]["overall"],results[0]["overall"])
        result=self.run_cli(*args,"--max-worst-harm","10","--max-tail-harm","10")
        self.assertEqual(result["decision"],"BLOCK")
        self.assertLess(result["evidence"]["unit_gain"],0)
        self.assertIn("unit_gain_below_minimum",result["evidence"]["reasons"])
        self.assertNotIn("overall_gain_below_minimum",result["evidence"]["reasons"])

    def test_secondary_thresholds_do_not_inherit_primary_scale(self):
        args,_,_,_=self.secondary_fixture()
        result=self.run_cli(*args,"--min-gain",".8","--ci-min-gain",".8")
        self.assertEqual(result["decision"],"PASS")
        self.assertEqual(result["evidence"]["unit_min_gain"],0)
        self.assertEqual(result["evidence"]["unit_ci_min_gain"],0)
        blocked=self.run_cli(*args,"--unit-min-gain",".8","--unit-ci-min-gain",".8")
        self.assertEqual(blocked["decision"],"BLOCK")
        self.assertIn("unit_gain_below_minimum",blocked["evidence"]["reasons"])
        self.assertIn("unit_ci_not_above_minimum",blocked["evidence"]["reasons"])
        self.assertNotIn("overall_ci_not_above_minimum",blocked["evidence"]["reasons"])

    def test_secondary_units_are_recomputed_from_rows(self):
        args,_,_,paths=self.secondary_fixture()
        records=analysis.csv_records(paths[1],["unit","fold","group","score"])
        records[0]["score"]="1"
        self.csvfile(paths[1].name,["unit","fold","group","score"],
            [[r["unit"],r["fold"],r["group"],r["score"]] for r in records])
        result=self.run_cli(*args,ok=False)
        self.assertIn("row/unit score mismatch",result["error"])

    def test_qwk_pooled_score_uses_explicit_single_unit_mae(self):
        args,results,rows,_=self.secondary_fixture(metric="qwk",unit_metric="mae")
        self.assertEqual((results[0]["overall"],results[1]["overall"]),(0,1))
        self.assertTrue(all(unit["score"] == 0 for unit in results[1]["units"]))
        self.run_cli("score","--rows",rows[1],"--metric","qwk",ok=False)
        result=self.run_cli(*args)
        self.assertEqual(result["decision"],"PASS")
        self.assertEqual(result["evidence"]["unit_metric"],{"name":"mae","direction":"lower"})

    def test_secondary_limits_must_be_finite_and_nonnegative(self):
        args,_,_,_=self.secondary_fixture()
        for flag,value in (("--unit-min-gain","-1"),("--unit-ci-min-gain","-1")):
            with self.subTest(flag=flag):
                self.run_cli(*args,flag,value,ok=False)

    def test_custom_cli(self):
        script=self.path/"scorer.py"
        script.write_text("def custom(y,p,w):\n    return sum(a*abs(b-c) for a,b,c in zip(w,y,p))/sum(w)\n")
        rows=self.csvfile("rows.csv",["row_id","unit","fold","y_true","y_pred"],[["a","x","0",0,2]])
        result=self.run_cli("score","--rows",rows,"--metric","custom","--scorer",str(script)+":custom")
        self.assertEqual(result["overall"],2)

    def test_duplicate_and_nonfinite_csv(self):
        for rows in [[["x",1],["x",2]],[["x","nan"]],[["x","inf"]],[["x",""]]]:
            with self.subTest(rows=rows):
                path=self.csvfile("bad.csv",["unit","score"],rows)
                self.run_cli("noise","--units",path,ok=False)

    def test_bad_headers_and_width(self):
        for content in ["unit,score,score\nx,1,2\n","unit,score\nx,1,2\n","unit,score\nx\n","unit,score\n"]:
            path=self.path/"bad.csv"; path.write_text(content)
            self.run_cli("noise","--units",path,ok=False)

    def test_duplicate_rows_and_spanning_fold(self):
        for rows in [[["a","x","0",0,0],["a","x","0",0,0]],[["a","x","0",0,0],["b","x","1",0,0]]]:
            path=self.csvfile("bad.csv",["row_id","unit","fold","y_true","y_pred"],rows)
            self.run_cli("score","--rows",path,"--metric","mae",ok=False)

    def test_unit_alignment_mismatches(self):
        left=analysis.read_units(self.units())
        for field,value in [("unit","other"),("weight",2),("fold","other"),("group","other")]:
            right=[dict(row) for row in left]; right[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                analysis.paired(left,right,"unit")

    def test_row_alignment_truth_mismatch(self):
        args=self.gate_fixture()
        path=self.path/"cand_rows.csv"
        text=path.read_text().replace("0,0,0,0,0,1","0,0,0,0,1,2")
        path.write_text(text)
        self.run_cli(*args,ok=False)

    def test_gate_pass_and_no_mutation(self):
        args=self.gate_fixture(); before=self.ledger_path.read_bytes()
        result=self.run_cli(*args,"--min-gain",".5")
        self.assertEqual(result["decision"],"PASS")
        self.assertEqual(result["evidence"]["unit_ci"],[1,1])
        self.assertEqual(result["evidence"]["overall"]["ci"],[1,1])
        self.assertEqual(before,self.ledger_path.read_bytes())
        self.assertEqual(result["artifact_sha256"],hashlib.sha256((self.path/"cand_rows.csv").read_bytes()).hexdigest())

    def test_gate_direction_higher(self):
        self.data["metric"]={"name":"custom","direction":"higher"}; self.save()
        script=self.path/"custom.py"; script.write_text("def scorer(y,p,w): return sum(a*b for a,b in zip(p,w))/sum(w)\n")
        args=self.gate_fixture([1]*6,[2]*6)
        result=self.run_cli(*args,"--scorer",str(script)+":scorer")
        self.assertEqual(result["decision"],"PASS")

    def test_gate_reject_harm_and_gain(self):
        result=self.run_cli(*self.gate_fixture([2]*6,[3]*6))
        self.assertEqual(result["decision"],"BLOCK")
        self.assertIn("worst_unit_harm",result["evidence"]["reasons"])
        self.assertIn("overall_ci_not_above_minimum",result["evidence"]["reasons"])

    def test_gate_min_gain_and_ci(self):
        result=self.run_cli(*self.gate_fixture(),"--min-gain","2","--ci-min-gain","1")
        self.assertIn("unit_gain_below_minimum",result["evidence"]["reasons"])
        self.assertIn("unit_ci_not_above_minimum",result["evidence"]["reasons"])

    def test_gate_tolerated_tail(self):
        args=self.gate_fixture([2]*6,[.5,.5,.5,.5,.5,2.1])
        strict=self.run_cli(*args)
        tolerant=self.run_cli(*args,"--max-worst-harm",".11","--max-tail-harm",".11")
        self.assertIn("worst_unit_harm",strict["evidence"]["reasons"])
        self.assertNotIn("worst_unit_harm",tolerant["evidence"]["reasons"])

    def test_gate_no_overall_is_blocked(self):
        args=self.gate_fixture(); idx=args.index("--baseline-rows")
        result=self.run_cli(*args[:idx],"--resamples","100")
        self.assertEqual(result["decision"],"BLOCK")
        self.assertIn("overall_rows_required",result["evidence"]["reasons"])

    def test_gate_validator_stage_and_blocker(self):
        self.data["validator"]={"authority":"veto_only","covered_stages":["data"]}
        self.data["findings"]=[{"id":"leak","severity":"blocker","status":"open"}]; self.save()
        result=self.run_cli(*self.gate_fixture())
        self.assertIn("validator_veto_only",result["evidence"]["reasons"])
        self.assertIn("unresolved_blockers:leak",result["evidence"]["reasons"])
        self.assertTrue(any(r.startswith("validator_missing_stages") for r in result["evidence"]["reasons"]))

    def test_gate_fold_group_limits(self):
        result=self.run_cli(*self.gate_fixture(),"--min-folds","4","--min-groups","3")
        self.assertIn("insufficient_folds",result["evidence"]["reasons"])
        self.assertIn("insufficient_groups",result["evidence"]["reasons"])

    def test_gate_fresh_replication_binds_artifact(self):
        args=self.gate_fixture()
        result=self.run_cli(*args,"--search-trials","2")
        self.assertIn("fresh_independent_replication_required",result["evidence"]["reasons"])
        receipt={"fresh":True,"independent":True,"evidence_id":"independent-resplit-001","candidate_sha256":hashlib.sha256((self.path/"cand_rows.csv").read_bytes()).hexdigest()}
        path=self.jsonfile("replication.json",receipt)
        self.assertEqual(self.run_cli(*args,"--search-trials","2","--replication",path)["decision"],"PASS")
        receipt["candidate_sha256"]="0"*64; self.jsonfile("replication.json",receipt)
        self.assertEqual(self.run_cli(*args,"--search-trials","2","--replication",path)["decision"],"BLOCK")

    def test_gate_cannot_reduce_registered_sweep_count(self):
        args=self.gate_fixture()
        candidate=self.candidate("search")
        candidate["swept"]=5
        candidate["artifact_sha256"]=hashlib.sha256((self.path/"cand_rows.csv").read_bytes()).hexdigest()
        self.data["candidates"]=[candidate];self.save()
        result=self.run_cli(*args,"--candidate-id","search","--search-trials","1")
        self.assertEqual(result["evidence"]["search_trials"],5)
        self.assertEqual(result["decision"],"BLOCK")

    def test_gate_registered_artifact_must_match(self):
        args=self.gate_fixture()
        self.data["candidates"]=[self.candidate("wrong-artifact")];self.save()
        result=self.run_cli(*args,"--candidate-id","wrong-artifact",ok=False)
        self.assertIn("artifact does not match",result["error"])

    def test_gate_deterministic_seed(self):
        args=self.gate_fixture([2]*6,[.2,.4,.6,.8,1,1.2])
        self.assertEqual(self.run_cli(*args),self.run_cli(*args))

    def test_unit_score_rows_must_recompute(self):
        args=self.gate_fixture(); path=self.path/"candidate.csv"
        path.write_text(path.read_text().replace("0,1,1,0,0","0,0.5,1,0,0"))
        self.assertIn("score mismatch",self.run_cli(*args,ok=False)["error"])

    def test_nonadditive_pooled_rmse(self):
        self.data["metric"]={"name":"rmse","direction":"lower"}; self.save()
        result=self.run_cli(*self.gate_fixture([2,3,4,5,6,7],[1,2,3,4,5,6]))
        self.assertNotEqual(result["evidence"]["unit_gain"],result["evidence"]["overall"]["gain"])

    def test_noise_and_simulated_prior(self):
        path=self.units()
        result=self.run_cli("noise","--units",path,"--cluster","group","--resamples","100","--simulated-sd","2","--simulated-n","16")
        self.assertEqual(result["clusters"],2)
        self.assertGreater(result["noise_floor"],0)
        self.assertEqual(result["simulation_prior"]["source"],"SIMULATED_PRIOR_NOT_OBSERVED_EVIDENCE")
        self.assertEqual(result["source"],"observed_absolute_unit_scores")

    def test_noise_one_cluster(self):
        path=self.csvfile("one.csv",["unit","score","group"],[["a",1,"x"],["b",2,"x"]])
        result=self.run_cli("noise","--units",path,"--cluster","group","--resamples","100")
        self.assertIsNone(result["noise_floor"])
        self.assertEqual(result["status"],"insufficient_independent_clusters")

    def test_noise_target_test_scale(self):
        path=self.units()
        small=self.run_cli("noise","--units",path,"--target-units","6","--resamples","400")
        large=self.run_cli("noise","--units",path,"--target-units","60","--resamples","400")
        self.assertEqual(large["target_independent_clusters"],60)
        self.assertLess(large["bootstrap_sd"],small["bootstrap_sd"])

    def test_prior_without_observed_units(self):
        result=self.run_cli("noise","--simulated-sd","2","--simulated-n","20","--resamples","100")
        self.assertEqual(result["status"],"no_observed_evidence")
        self.assertNotIn("mean",result)

    def test_binary_and_ordinal_simulated_priors(self):
        for metric in ["accuracy","f1","logloss","auc","qwk"]:
            with self.subTest(metric=metric):
                args=["noise","--prior-metric",metric,"--target-units","4","--rows-per-unit","8","--effect","1","--correlation",".4","--resamples","100"]
                result=self.run_cli(*args)
                prior=result["metric_simulation_prior"]
                self.assertEqual(prior["source"],"SIMULATED_ASSUMPTIONS_NOT_OBSERVED_EVIDENCE")
                self.assertEqual(prior["assumptions"]["within_unit_latent_noise_correlation"],.4)
                self.assertEqual(prior["valid_draws"]+prior["invalid_draws"],100)
                self.assertEqual(prior,self.run_cli(*args)["metric_simulation_prior"])

    def test_invalid_prior_parameters(self):
        for flag,value in [("--prevalence","1"),("--correlation","1"),("--effect","-1"),("--rows-per-unit","1")]:
            self.run_cli("noise","--prior-metric","accuracy","--target-units","3","--resamples","100",flag,value,ok=False)

    def test_group_report(self):
        result=self.run_cli("groups","--units",self.units(),"--resamples","100")
        self.assertEqual(len(result["groups"]),2)
        self.assertEqual(result["groups"][0]["units"],3)
        self.assertEqual(result["overall"]["clusters"],2)

    def test_groups_compare_rank_and_sign_flip(self):
        a=self.units("a.csv",[1,3,1,3,1,3])
        b=self.units("b.csv",[2,0,2,0,2,0])
        result=self.run_cli("groups","--candidate",f"a={a}","--candidate",f"b={b}","--resamples","100")
        self.assertEqual(result["group_rankings"]["0"][0]["candidate"],"a")
        self.assertEqual(result["group_rankings"]["1"][0]["candidate"],"b")
        self.assertEqual(result["sign_flips"][0]["group"],"0")

    def test_groups_comparison_requires_same_units(self):
        a=self.units("a.csv")
        b=self.csvfile("b.csv",["unit","score","fold","group"],[["new",1,"0","0"]])
        self.run_cli("groups","--candidate",f"a={a}","--candidate",f"b={b}","--resamples","100",ok=False)

    def test_null_shuffle(self):
        path=self.csvfile("null.csv",["row_id","unit","fold","y_true","y_pred"],[[str(i),"a","0",i%2,i%2] for i in range(10)])
        result=self.run_cli("null","--rows",path,"--metric","accuracy","--resamples","100")
        self.assertEqual(result["observed"],1)
        self.assertLess(result["null_mean"],1)
        self.assertGreater(result["p_value"],0)
        self.assertEqual(result["shuffle_within"],["unit","fold"])

    def test_null_constant_insufficient(self):
        path=self.csvfile("null.csv",["row_id","unit","fold","y_true","y_pred"],[["a","a","0",0,1],["b","a","0",0,1]])
        result=self.run_cli("null","--rows",path,"--metric","mae","--resamples","100")
        self.assertEqual(result["status"],"insufficient_exchangeable_variation")

    def test_spearman_ties_constant_small(self):
        self.assertEqual(analysis.correlation([1,2,2,3],[1,2,2,3])["rho"],1)
        self.assertEqual(analysis.correlation([1,1,1],[1,2,3])["status"],"constant_scores")
        self.assertEqual(analysis.correlation([1],[2])["status"],"insufficient_samples")

    def trust_candidates(self):
        self.data["candidates"]=[self.candidate(str(i),i,root="family") for i in range(8)]
        self.save()

    def test_trust_all_three_views(self):
        self.trust_candidates()
        result=self.run_cli("trust","--ledger",self.ledger_path)
        self.assertEqual(result["regime"],"ALIGNED")
        self.assertEqual(result["overall"]["rho"],1)
        self.assertEqual(result["structural"]["rho"],1)
        self.assertEqual(result["within_family"]["family"]["rho"],1)
        self.assertEqual(len(result["candidate_ids"]),8)
        self.assertEqual(result["n_paired"],8)
        self.assertIn("computed_at",result)

    def test_trust_public_unit_limit(self):
        self.trust_candidates(); self.data["trust"]["public_units"]=2; self.save()
        self.assertEqual(self.run_cli("trust","--ledger",self.ledger_path)["regime"],"INSUFFICIENT")

    def test_trust_within_family_cannot_be_hidden_by_overall(self):
        self.trust_candidates()
        for c in self.data["candidates"]: c["family"]=c["id"]
        self.save()
        result=self.run_cli("trust","--ledger",self.ledger_path)
        self.assertEqual(result["overall"]["rho"],1)
        self.assertEqual(result["regime"],"INSUFFICIENT")

    def test_trust_divergent(self):
        self.trust_candidates()
        for c in self.data["candidates"]: c["public"]=-c["oof"]
        self.save()
        self.assertEqual(self.run_cli("trust","--ledger",self.ledger_path)["regime"],"DIVERGENT")

    def test_leaderboard_explicit_thresholds_ties(self):
        table=self.csvfile("lb.csv",["id","score"],[["a",.9],["b",.9],["c",.8]])
        thresholds=self.jsonfile("medals.json",{"gold":.9,"silver":.8})
        result=self.run_cli("leaderboard","--table",table,"--thresholds",thresholds,"--direction","higher")
        self.assertEqual(result["platforms"][0]["ids"],["a","b"])
        self.assertEqual(result["platforms"][0]["rank_end"],2)
        self.assertEqual(result["platforms"][1]["medals_meeting_score_threshold"],["silver"])
        self.assertFalse(result["official_awards_inferred"])

    def test_select_public_never_overrides_local(self):
        a,b=self.candidate("a",1),self.candidate("b",2)
        a["public"],b["public"]=100,0
        self.data["candidates"]=[a,b]; self.save()
        result=self.run_cli("select","--ledger",self.ledger_path,"--count","1")
        self.assertEqual(result["recommendations"][0]["id"],"a")
        self.assertEqual(result["authority"],"oof")

    def test_select_requires_retrieved_public_and_fresh_replication(self):
        a,b=self.candidate("a",1),self.candidate("b",2)
        a["public"]=None;b["swept"]=3
        self.data["candidates"]=[a,b];self.save()
        result=self.run_cli("select","--ledger",self.ledger_path)
        self.assertEqual(result["candidate_ids"],[])
        self.assertIn("public_score_not_retrieved",result["excluded"]["a"])
        self.assertIn("fresh_replication_or_search_count_missing",result["excluded"]["b"])

    def test_select_gate_metric_must_match_current_ledger(self):
        for metric in [None,{"name":"accuracy","direction":"higher"}]:
            a=self.candidate("a")
            a["gate"]["evidence"]["metric"]=metric
            self.data["candidates"]=[a];self.save()
            result=self.run_cli("select","--ledger",self.ledger_path)
            self.assertEqual(result["candidate_ids"],[])
            self.assertIn("gate_metric_stale_or_missing",result["excluded"]["a"])

    def test_select_holdout_authority_and_missing_score(self):
        self.data["holdout"]["opened_at"]="2026-01-01T00:00:00Z"
        a,b,c=self.candidate("a",1),self.candidate("b",2),self.candidate("c",0)
        a["holdout"],b["holdout"]=2,1
        self.data["candidates"]=[a,b,c];self.save()
        result=self.run_cli("select","--ledger",self.ledger_path,"--count","1")
        self.assertEqual(result["recommendations"][0]["id"],"b")
        self.assertIn("missing_authoritative_score",result["excluded"]["c"])

    def test_select_stale_artifact_blocker_validator(self):
        a=self.candidate("a"); a["artifact_sha256"]="f"*64
        self.data["candidates"]=[a];self.save()
        result=self.run_cli("select","--ledger",self.ledger_path)
        self.assertEqual(result["recommendations"],[])
        self.assertIn("gate_artifact_identity_stale_or_missing",result["excluded"]["a"])
        self.data["candidates"]=[self.candidate("a")]
        self.data["findings"]=[{"id":"x","severity":"blocker","status":"open"}];self.save()
        self.assertEqual(self.run_cli("select","--ledger",self.ledger_path)["status"],"blocked")
        self.data["findings"]=[];self.data["validator"]["authority"]="veto_only";self.save()
        self.assertEqual(self.run_cli("select","--ledger",self.ledger_path)["status"],"blocked")

    def test_selection_requires_residual_baseline(self):
        self.data["candidates"]=[self.candidate("a",1),self.candidate("b",2)];self.save()
        result=self.run_cli("select","--ledger",self.ledger_path)
        self.assertEqual(len(result["recommendations"]),1)
        self.assertEqual(result["diversity"][1]["checks"][0]["status"],"residual_baseline_or_unit_alignment_missing")

    def test_selection_residual_and_root_diversity(self):
        a=self.candidate("a",1,units={"a":9,"b":10,"c":11,"d":12})
        b=self.candidate("b",2,units={"a":12,"b":11,"c":10,"d":9})
        c=self.candidate("c",1.5,root="a",units=b["selection_units"])
        self.data["candidates"]=[a,b,c];self.save()
        baseline=self.csvfile("residual.csv",["unit","score"],[[key,10] for key in "abcd"])
        result=self.run_cli("select","--ledger",self.ledger_path,"--residual-baseline",baseline)
        self.assertEqual([r["id"] for r in result["recommendations"]],["a","b"])
        self.assertIn("c",result["excluded"])
        self.assertEqual(result["diversity"][-1]["checks"][0]["correlation"],-1)

    def hedge_fixture(self):
        a=self.candidate("a",1,root="family",units={"a":9,"b":10,"c":11,"d":12})
        b=self.candidate("b",1.1,root="b",units={"a":12,"b":11,"c":10,"d":9})
        c=self.candidate("c",1.2,root="c",units={"a":11,"b":9,"c":12,"d":10})
        d=self.candidate("d",1.3,root="d",units={"a":12,"b":11,"c":10,"d":9})
        for candidate,value in zip([a,b,c,d],[1,1.2,1.1,1.3]):
            candidate["family"]="family"
            candidate["public"]=value
        self.data["candidates"]=[a,b,c,d];self.save()
        trust=self.run_cli("trust","--ledger",self.ledger_path)
        self.assertEqual(trust["regime"],"ALIGNED")
        self.data["trust"]=trust;self.save()
        baseline=self.csvfile("residual.csv",["unit","score"],[[key,10] for key in "abcd"])
        return ["select","--ledger",self.ledger_path,"--residual-baseline",baseline,"--public-second","--public-max-local-regret",".3"]

    def test_select_fresh_public_diverse_second(self):
        result=self.run_cli(*self.hedge_fixture())
        self.assertEqual(result["candidate_ids"],["a","c"])
        self.assertEqual(result["public_role"],"diverse_second_with_local_regret_limit")

    def test_select_public_requires_current_fingerprint(self):
        args=self.hedge_fixture()
        self.data["candidates"][2]["public"]=.5;self.save()
        result=self.run_cli(*args)
        self.assertEqual(result["candidate_ids"],["a","b"])
        self.assertIn("trust_fingerprint_stale_or_missing",result["public_hedge_reasons"])

    def test_select_public_requires_current_time_and_units(self):
        args=self.hedge_fixture()
        self.data["trust"]["computed_at"]="2000-01-01T00:00:00Z";self.data["trust"]["public_units"]=1;self.save()
        result=self.run_cli(*args)
        self.assertEqual(result["candidate_ids"],["a","b"])
        self.assertIn("trust_timestamp_stale",result["public_hedge_reasons"])
        self.assertIn("public_units_insufficient",result["public_hedge_reasons"])

    def test_select_public_local_credibility_tolerance(self):
        args=self.hedge_fixture()
        result=self.run_cli(*args,"--public-max-local-regret",".1")
        self.assertEqual(result["candidate_ids"],["a","b"])

    def test_select_insufficient_public_allows_local_promoted(self):
        candidate=self.candidate("a");candidate["status"]="promoted"
        self.data["candidates"]=[candidate];self.save()
        result=self.run_cli("select","--ledger",self.ledger_path,"--public-second")
        self.assertEqual(result["candidate_ids"],["a"])
        self.assertEqual(result["status"],"recommendation_only")

    def test_postmortem_best_unselected(self):
        a,b,c=[self.candidate(name,n) for n,name in enumerate("abc")]
        for item,value in zip([a,b,c],[3,1,2]): item["private"]=value
        a["status"]="selected";self.data["candidates"]=[a,b,c];self.save()
        result=self.run_cli("postmortem","--ledger",self.ledger_path,"--min-samples","3")
        self.assertEqual(result["best_unselected"]["id"],"b")
        self.assertEqual(result["missed_gain"],2)
        self.assertEqual(result["public_private_spearman"]["n"],3)

    def test_postmortem_signed_target_gap_decomposition(self):
        a,b=self.candidate("a"),self.candidate("b")
        a["private"],b["private"]=3,1
        a["status"]="selected";self.data["candidates"]=[a,b];self.save()
        result=self.run_cli("postmortem","--ledger",self.ledger_path,"--target-score","0")
        gaps=result["gap_decomposition"]
        self.assertEqual(gaps["model_frontier_gap"],1)
        self.assertEqual(gaps["selection_regret"],2)
        self.assertEqual(gaps["total_gap"],3)
        self.assertEqual(gaps["best_evaluated_id"],"b")

    def test_json_duplicate_and_nan(self):
        for text in ['{"metric":{},"metric":{}}','{"x":NaN}','{"x":1e999}']:
            path=self.path/"bad.json";path.write_text(text)
            with self.assertRaises(ValueError): analysis.read_json(path)

    def test_ledger_duplicate_candidates_and_invalid_score(self):
        a=self.candidate("a");self.data["candidates"]=[a,a];self.save()
        self.run_cli("select","--ledger",self.ledger_path,ok=False)
        self.data["candidates"]=[self.candidate("a")];self.data["candidates"][0]["oof"]="NaN";self.save()
        self.run_cli("select","--ledger",self.ledger_path,ok=False)

    def test_null_trust_is_not_a_local_selection_blocker(self):
        self.data["trust"]=None
        self.data["candidates"]=[self.candidate("a")];self.save()
        result=self.run_cli("select","--ledger",self.ledger_path,"--public-second")
        self.assertEqual(result["candidate_ids"],["a"])

    def test_help_all_commands(self):
        for name in ["score","gate","noise","groups","null","trust","leaderboard","select","postmortem"]:
            completed=subprocess.run([sys.executable,str(SCRIPTS/"analysis.py"),name,"--help"],capture_output=True,text=True)
            self.assertEqual(completed.returncode,0)
            self.assertIn("usage:",completed.stdout)


if __name__ == "__main__":
    unittest.main()
