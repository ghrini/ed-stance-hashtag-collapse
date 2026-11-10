"""Two things a reviewer flagged as unpersisted: (1) the repair's per-seed
stability numbers, recomputed directly from the CANONICAL thresholds saved
by recalibration_recovery.py (not reimplemented from scratch, to avoid any
drift); (2) the Saerens EM label-shift control, properly saved this time
(it was previously run as a throwaway scratch script and never persisted).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

P = Path(__file__).resolve().parent.parent
SEEDS = [13, 42, 71, 123, 2026]

# ---------------------------------------------------------------- part 1
recal = json.load(open(P / "v2" / "RECALIBRATION_RESULTS.json"))
thr = {int(k): v for k, v in recal["recalibrated_thresholds_by_fold"].items()}

per_seed = {}
for seed in SEEDS:
    parts = [pd.read_csv(f) for f in
             glob.glob(str(P / "runs" / "independent" / "fold_*" / f"seed_{seed}" / "predictions_nohash.csv"))]
    d = pd.concat(parts).sort_values("doc_id").reset_index(drop=True)
    y = d.label_promoting.to_numpy(int)
    f_frozen = f1_score(y, d.pred_promoting, zero_division=0)
    d["recal_thr"] = d.outer_fold.map(thr)
    f_recal = f1_score(y, (d.prob_promoting >= d.recal_thr).astype(int), zero_division=0)
    per_seed[seed] = {"frozen": float(f_frozen), "recal": float(f_recal), "delta": float(f_recal - f_frozen)}

deltas = np.array([v["delta"] for v in per_seed.values()])
frozen_vals = np.array([v["frozen"] for v in per_seed.values()])
recal_vals = np.array([v["recal"] for v in per_seed.values()])
from scipy import stats as st
m, sd = deltas.mean(), deltas.std(ddof=1)
lo, hi = st.t.interval(0.95, 4, loc=m, scale=sd / np.sqrt(5))

part1 = {
    "note": "Per-seed frozen->recalibrated promotion F1, using the CANONICAL "
            "per-fold thresholds from RECALIBRATION_RESULTS.json (not reimplemented).",
    "per_seed": per_seed,
    "delta_min": float(deltas.min()), "delta_max": float(deltas.max()),
    "delta_mean": float(m), "delta_seed_t_ci": [float(lo), float(hi)],
    "frozen_spread": float(frozen_vals.max() - frozen_vals.min()),
    "recal_spread": float(recal_vals.max() - recal_vals.min()),
}

# ---------------------------------------------------------------- part 2
core = pd.read_csv(P / "data" / "core_clean_deduped.csv")
manifest = pd.read_csv(P / "artifacts" / "dev_outer_folds_v1.csv")
core = core.merge(manifest[["doc_id", "outer_fold"]], left_on="Column1", right_on="doc_id")
test = pd.read_csv(P / "v2" / "final" / "baseline_nohash_5seed.csv").sort_values("doc_id").reset_index(drop=True)
y = test.label_promoting.to_numpy(int)

preds_em = np.zeros(len(test), dtype=int)
est_priors = {}
for fold in range(5):
    pi0 = float(core[core.outer_fold != fold].label_promoting.mean())
    m_ = test.outer_fold == fold
    p = test.loc[m_, "prob_promoting"].to_numpy()
    pi = pi0
    for _ in range(200):
        w = (pi / pi0) * p / ((pi / pi0) * p + ((1 - pi) / (1 - pi0)) * (1 - p))
        pi_new = float(w.mean())
        if abs(pi_new - pi) < 1e-8:
            break
        pi = pi_new
    est_priors[fold] = {"train_prior": pi0, "em_estimated_prior": pi, "true_test_prevalence": 0.236}
    w = (pi / pi0) * p / ((pi / pi0) * p + ((1 - pi) / (1 - pi0)) * (1 - p))
    preds_em[m_.to_numpy()] = (w >= 0.5).astype(int)

f1_saerens = float(f1_score(y, preds_em, zero_division=0))
f1_frozen = float(f1_score(y, test.pred_promoting, zero_division=0))
part2 = {
    "note": "Saerens et al. (2002) EM prior-correction control: assumes label "
            "shift (p(x|y) fixed, prior moves). Our shift is the opposite "
            "(inputs moved, prior did not), so this should misfire.",
    "est_priors_by_fold": est_priors,
    "saerens_em_f1": f1_saerens,
    "frozen_f1_reference": f1_frozen,
}

out = {"per_seed_stability": part1, "saerens_em_control": part2}
out_path = Path(__file__).with_name("REPAIR_ROBUSTNESS_RESULTS.json")
out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
