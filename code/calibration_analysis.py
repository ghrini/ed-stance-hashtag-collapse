"""Calibration-controlled robustness analysis.

The v1 diagnostic compared hashtag-stripped F1 using thresholds tuned on
CLEAN validation data. That conflates two effects: a genuine loss of
discriminative signal, and a pure calibration shift (stripped probabilities
move relative to a frozen threshold). This script separates them:

  1. threshold-free ranking quality (AUPRC), and
  2. F1 at each system's own oracle threshold on the stripped data,

and reports the frozen-threshold number as what it actually is - a measure of
calibration brittleness under input perturbation, not of shortcut reliance.

Vectorised best-F1 over a probability ranking so bootstraps are cheap.
Writes v2/CALIBRATION_RESULTS.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
F = HERE / "final"
NB = 10000
SEED = 20260818


def best_f1_fast(y, p):
    """Max F1 over all thresholds, via a single sort (no threshold grid)."""
    order = np.argsort(-p, kind="stable")
    ys = y[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    total_pos = ys.sum()
    if total_pos == 0:
        return 0.0
    f1 = 2 * tp / (k + total_pos)
    return float(f1.max())


def load():
    b_nh = pd.read_csv(F / "baseline_nohash_5seed.csv").sort_values("doc_id").reset_index(drop=True)
    h_nh = pd.read_csv(F / "htdrop_nohash_5seed.csv").sort_values("doc_id").reset_index(drop=True)
    b_cl = pd.read_csv(PROJECT / "runs" / "independent" / "aggregate" / "oof_ensemble_nested.csv"
                       ).sort_values("doc_id").reset_index(drop=True)
    h_cl = pd.read_csv(F / "htdrop5" / "oof_ensemble_nested.csv").sort_values("doc_id").reset_index(drop=True)
    for a, c in [(b_nh, b_cl), (h_nh, h_cl), (b_nh, h_nh)]:
        assert (a.doc_id.values == c.doc_id.values).all()
        assert (a.label_promoting.values == c.label_promoting.values).all()
    return b_nh, h_nh, b_cl, h_cl


def main():
    b_nh, h_nh, b_cl, h_cl = load()
    y = b_nh.label_promoting.to_numpy(int)
    ys = b_nh.label_suffering.to_numpy(int)
    strata = [g.index.to_numpy() for _, g in b_nh.groupby(["outer_fold", "stream_group"], sort=True)]
    rng = np.random.default_rng(SEED)
    boot_idx = [np.concatenate([rng.choice(s, len(s), replace=True) for s in strata]) for _ in range(NB)]

    P = {"base_clean": b_cl.prob_promoting.to_numpy(), "base_strip": b_nh.prob_promoting.to_numpy(),
         "htd_clean": h_cl.prob_promoting.to_numpy(), "htd_strip": h_nh.prob_promoting.to_numpy()}
    D = {"base_strip": b_nh.pred_promoting.to_numpy(int), "htd_strip": h_nh.pred_promoting.to_numpy(int)}

    def ci(fn):
        obs = fn(np.arange(len(y)))
        vals = np.fromiter((fn(i) for i in boot_idx), float, NB)
        return {"estimate": float(obs), "ci_low": float(np.quantile(vals, .025)),
                "ci_high": float(np.quantile(vals, .975))}

    out = {"note": "Promotion label. 'strip' = all hashtags removed from test text. "
                   "10,000 stratified bootstrap replicates within outer_fold x stream_group.",
           "point_estimates": {}, "inference": {}}

    pe = out["point_estimates"]
    for k, p in P.items():
        pe[k] = {"auprc": float(average_precision_score(y, p)), "best_f1": best_f1_fast(y, p)}
    pe["base_strip"]["f1_at_frozen_clean_threshold"] = float(f1_score(y, D["base_strip"], zero_division=0))
    pe["htd_strip"]["f1_at_frozen_clean_threshold"] = float(f1_score(y, D["htd_strip"], zero_division=0))
    pe["base_clean"]["f1_at_own_threshold"] = float(f1_score(y, b_cl.pred_promoting, zero_division=0))
    pe["htd_clean"]["f1_at_own_threshold"] = float(f1_score(y, h_cl.pred_promoting, zero_division=0))

    inf = out["inference"]
    # (A) Is the shortcut real, threshold-free? Baseline clean vs stripped.
    inf["shortcut_baseline_auprc_clean_minus_strip"] = ci(
        lambda i: average_precision_score(y[i], P["base_clean"][i]) - average_precision_score(y[i], P["base_strip"][i]))
    inf["shortcut_baseline_bestf1_clean_minus_strip"] = ci(
        lambda i: best_f1_fast(y[i], P["base_clean"][i]) - best_f1_fast(y[i], P["base_strip"][i]))
    # (B) Does hashtag dropout help at matched calibration?
    inf["htdrop_vs_base_strip_auprc"] = ci(
        lambda i: average_precision_score(y[i], P["htd_strip"][i]) - average_precision_score(y[i], P["base_strip"][i]))
    inf["htdrop_vs_base_strip_bestf1"] = ci(
        lambda i: best_f1_fast(y[i], P["htd_strip"][i]) - best_f1_fast(y[i], P["base_strip"][i]))
    # (C) The frozen-threshold gap = calibration brittleness, not discrimination.
    inf["htdrop_vs_base_strip_f1_frozen_thresholds"] = ci(
        lambda i: f1_score(y[i], D["htd_strip"][i], zero_division=0) - f1_score(y[i], D["base_strip"][i], zero_division=0))
    # (D) Cost on clean inputs, threshold-free.
    inf["htdrop_vs_base_clean_auprc"] = ci(
        lambda i: average_precision_score(y[i], P["htd_clean"][i]) - average_precision_score(y[i], P["base_clean"][i]))

    # Decomposition of the frozen-threshold collapse
    base_frozen = pe["base_strip"]["f1_at_frozen_clean_threshold"]
    base_oracle = pe["base_strip"]["best_f1"]
    base_clean_oracle = pe["base_clean"]["best_f1"]
    out["decomposition"] = {
        "baseline_clean_oracle_f1": base_clean_oracle,
        "baseline_strip_oracle_f1": base_oracle,
        "baseline_strip_frozen_f1": base_frozen,
        "loss_from_calibration_only": base_oracle - base_frozen,
        "loss_from_discrimination_only": base_clean_oracle - base_oracle,
        "fraction_of_collapse_due_to_calibration":
            (base_oracle - base_frozen) / (base_clean_oracle - base_frozen),
    }

    # Per-seed stability of the frozen-threshold delta (the unstable quantity)
    import glob
    per_seed = {}
    for seed in [13, 42, 71, 123, 2026]:
        rows = {}
        for tag, runs, system in [("base", PROJECT / "runs", "independent"),
                                  ("htd", PROJECT / "runs_v2", "bertweet_htdrop")]:
            parts = []
            for f in glob.glob(str(Path(runs) / system / "fold_*" / f"seed_{seed}" / "predictions_nohash.csv")):
                parts.append(pd.read_csv(f))
            d = pd.concat(parts).sort_values("doc_id")
            rows[tag] = f1_score(d.label_promoting, d.pred_promoting, zero_division=0)
        per_seed[seed] = {"baseline": rows["base"], "htdrop": rows["htd"],
                          "delta": rows["htd"] - rows["base"]}
    out["per_seed_frozen_threshold_delta"] = per_seed
    deltas = [v["delta"] for v in per_seed.values()]
    out["per_seed_summary"] = {"min": min(deltas), "max": max(deltas), "mean": float(np.mean(deltas))}

    # Suffering label, for completeness
    out["suffering_strip"] = {
        "baseline_auprc": float(average_precision_score(ys, b_nh.prob_suffering)),
        "htdrop_auprc": float(average_precision_score(ys, h_nh.prob_suffering)),
        "baseline_clean_auprc": float(average_precision_score(ys, b_cl.prob_suffering)),
        "htdrop_clean_auprc": float(average_precision_score(ys, h_cl.prob_suffering)),
    }

    (HERE / "CALIBRATION_RESULTS.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"point_estimates": pe, "inference": inf,
                      "decomposition": out["decomposition"],
                      "per_seed_summary": out["per_seed_summary"]}, indent=2))


if __name__ == "__main__":
    main()
