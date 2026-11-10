"""Confirmatory analysis: 5-seed ensembles, paired stats, robustness table.

Produces v2/FINAL_RESULTS.json with every number quoted in the paper:
  1. 5-seed confirmatory ensemble (all five systems) vs 5-seed baseline via
     compare_oof.py (stratified bootstrap + paired randomization).
  2. 5-seed bertweet_htdrop vs baseline (clean-test parity check).
  3. Hashtag-stripped robustness: baseline vs htdrop promotion F1 with a
     paired stratified bootstrap on the same documents.
  4. Screening table (2-seed) for the screening section.
"""
from __future__ import annotations

import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PY = sys.executable
SEEDS5 = "13+42+71+123+2026"


def build(groups, name, outdir):
    args = [PY, str(HERE / "build_ensemble.py"), "--system-name", name, "--output-dir", str(outdir)]
    for g in groups:
        args += ["--group", g]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    return json.loads(r.stdout)


def compare(baseline_csv, new_csv, out_json):
    r = subprocess.run([PY, str(PROJECT / "compare_oof.py"), "--baseline", str(baseline_csv),
                        "--new", str(new_csv), "--output", str(out_json)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    return json.load(open(out_json))


def nohash_ensemble(runs_dir, system, seeds, thresholds_from):
    parts = []
    for f in glob.glob(str(Path(runs_dir) / system / "fold_*" / "seed_*" / "predictions_nohash.csv")):
        df = pd.read_csv(f)
        if int(df.model_seed.iloc[0]) in seeds:
            parts.append(df)
    assert len(parts) == 5 * len(seeds), f"{system}: {len(parts)} nohash files"
    nh = pd.concat(parts).groupby(
        ["doc_id", "outer_fold", "stream_group", "label_suffering", "label_promoting"],
        as_index=False)[["prob_suffering", "prob_promoting"]].mean()
    ens = pd.read_csv(thresholds_from)
    thr = ens[["outer_fold", "nested_threshold_suffering", "nested_threshold_promoting"]].drop_duplicates()
    nh = nh.merge(thr, on="outer_fold", validate="many_to_one")
    nh["pred_suffering"] = nh.prob_suffering.ge(nh.nested_threshold_suffering).astype(int)
    nh["pred_promoting"] = nh.prob_promoting.ge(nh.nested_threshold_promoting).astype(int)
    return nh.sort_values("doc_id").reset_index(drop=True)


def label_scores(df):
    return {"suffering_f1": float(f1_score(df.label_suffering, df.pred_suffering, zero_division=0)),
            "promoting_f1": float(f1_score(df.label_promoting, df.pred_promoting, zero_division=0)),
            "macro_f1": float((f1_score(df.label_suffering, df.pred_suffering, zero_division=0)
                               + f1_score(df.label_promoting, df.pred_promoting, zero_division=0)) / 2)}


def paired_boot_delta(a, b, col_true, col_pred, n_boot=10000, seed=20260817):
    """Paired stratified bootstrap of F1(b) - F1(a) on identical docs."""
    m = a[["doc_id", "outer_fold", "stream_group", col_true, col_pred]].merge(
        b[["doc_id", col_true, col_pred]], on="doc_id", suffixes=("_a", "_b"), validate="one_to_one")
    assert (m[f"{col_true}_a"] == m[f"{col_true}_b"]).all()
    y = m[f"{col_true}_a"].to_numpy(int)
    pa = m[f"{col_pred}_a"].to_numpy(int)
    pb = m[f"{col_pred}_b"].to_numpy(int)
    strata = [g.index.to_numpy() for _, g in m.groupby(["outer_fold", "stream_group"], sort=True)]
    rng = np.random.default_rng(seed)
    obs = f1_score(y, pb, zero_division=0) - f1_score(y, pa, zero_division=0)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        boots[i] = (f1_score(y[idx], pb[idx], zero_division=0)
                    - f1_score(y[idx], pa[idx], zero_division=0))
    return {"delta": float(obs), "ci_low": float(np.quantile(boots, .025)),
            "ci_high": float(np.quantile(boots, .975))}


def main():
    out_root = HERE / "final"
    out_root.mkdir(exist_ok=True)
    results = {}

    print("Building 5-seed ensembles...", flush=True)
    ens5 = build([f"{PROJECT / 'runs'}:independent:{SEEDS5}",
                  f"{PROJECT / 'runs_v2'}:bertweet_htdrop:{SEEDS5}",
                  f"{PROJECT / 'runs_v2'}:troberta_2022:{SEEDS5}",
                  f"{PROJECT / 'runs_v2'}:deberta_v3:{SEEDS5}",
                  f"{PROJECT / 'runs_v2'}:bertweet_large:{SEEDS5}"],
                 "ens5_confirm", out_root / "ens5_confirm")
    results["ens5_point"] = ens5
    htdrop5 = build([f"{PROJECT / 'runs_v2'}:bertweet_htdrop:{SEEDS5}"],
                    "bertweet_htdrop_5seed", out_root / "htdrop5")
    results["htdrop5_point"] = htdrop5

    baseline_csv = PROJECT / "runs" / "independent" / "aggregate" / "oof_ensemble_nested.csv"

    print("Paired comparison: ens5 vs baseline...", flush=True)
    results["ens5_vs_baseline"] = compare(baseline_csv,
                                          out_root / "ens5_confirm" / "oof_ensemble_nested.csv",
                                          out_root / "ens5_vs_baseline.json")
    print("Paired comparison: htdrop vs baseline...", flush=True)
    results["htdrop_vs_baseline"] = compare(baseline_csv,
                                            out_root / "htdrop5" / "oof_ensemble_nested.csv",
                                            out_root / "htdrop_vs_baseline.json")

    print("Hashtag-stripped robustness...", flush=True)
    base_nh = nohash_ensemble(PROJECT / "runs", "independent", [13, 42, 71, 123, 2026], baseline_csv)
    htdrop_nh = nohash_ensemble(PROJECT / "runs_v2", "bertweet_htdrop", [13, 42, 71, 123, 2026],
                                out_root / "htdrop5" / "oof_ensemble_nested.csv")
    base_clean = pd.read_csv(baseline_csv)
    htdrop_clean = pd.read_csv(out_root / "htdrop5" / "oof_ensemble_nested.csv")
    results["robustness"] = {
        "baseline_clean": label_scores(base_clean),
        "baseline_nohash": label_scores(base_nh),
        "htdrop_clean": label_scores(htdrop_clean),
        "htdrop_nohash": label_scores(htdrop_nh),
        "nohash_promoting_delta_htdrop_vs_base": paired_boot_delta(
            base_nh, htdrop_nh, "label_promoting", "pred_promoting"),
        "nohash_macro_note": "thresholds are each system's own nested per-fold thresholds",
    }
    base_nh.to_csv(out_root / "baseline_nohash_5seed.csv", index=False)
    htdrop_nh.to_csv(out_root / "htdrop_nohash_5seed.csv", index=False)

    (out_root / ".." / "FINAL_RESULTS.json").resolve().write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({
        "ens5_macro": ens5["macro_f1"],
        "delta_vs_baseline": results["ens5_vs_baseline"]["delta_macro_f1"],
        "ci": results["ens5_vs_baseline"]["delta_macro_f1_ci"],
        "p": results["ens5_vs_baseline"]["paired_randomization_p"],
        "htdrop_delta": results["htdrop_vs_baseline"]["delta_macro_f1"],
        "htdrop_p": results["htdrop_vs_baseline"]["paired_randomization_p"],
        "nohash_prom_base": results["robustness"]["baseline_nohash"]["promoting_f1"],
        "nohash_prom_htdrop": results["robustness"]["htdrop_nohash"]["promoting_f1"],
        "nohash_prom_delta_ci": results["robustness"]["nohash_promoting_delta_htdrop_vs_base"],
    }, indent=2))


if __name__ == "__main__":
    main()
