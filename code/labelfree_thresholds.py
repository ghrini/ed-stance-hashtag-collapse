"""Label-free threshold adaptation under hashtag ablation.

Compares three rules that need NO labels on the shifted data, against the
frozen threshold (lower anchor), labeled recalibration (0.759), and the
oracle ceiling (0.784):

  A. Rate matching via stripped VALIDATION scores: keep the flag rate the
     model had on clean validation at its frozen threshold, by taking the
     matching quantile of the stripped validation score distribution.
     (Needs unlabeled ablated validation text only.)
  B. Rate matching via stripped TEST scores (transductive/monitoring style):
     same rule, quantile taken on the unlabeled test scores themselves.
  C. CDF mapping: map each stripped test score to its quantile in the
     stripped-val distribution, then to the clean-val score at that same
     quantile; apply the ORIGINAL frozen threshold to the mapped scores.

All evaluated on the same stripped outer-test predictions used everywhere
else (v2/final/baseline_nohash_5seed.csv), promotion label.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

P = Path(__file__).resolve().parent.parent
V2 = Path(__file__).resolve().parent

val = pd.read_csv(V2 / "baseline_val_preds_clean_and_stripped.csv")
test = pd.read_csv(P / "v2" / "final" / "baseline_nohash_5seed.csv")
frozen_thr = test[["outer_fold", "nested_threshold_promoting"]].drop_duplicates().set_index(
    "outer_fold").nested_threshold_promoting.to_dict()

rows = []
mapped_parts = []
for fold in range(5):
    vc = val[(val.outer_fold == fold) & (val.view == "clean")].sort_values("doc_id")
    vs = val[(val.outer_fold == fold) & (val.view == "stripped")].sort_values("doc_id")
    ts = test[test.outer_fold == fold]
    t_frozen = frozen_thr[fold]

    # clean-validation flag rate at frozen threshold
    r = float((vc.prob_promoting >= t_frozen).mean())

    # A: quantile of stripped-val scores
    t_a = float(np.quantile(vs.prob_promoting, 1 - r)) if r > 0 else 1.0
    # B: quantile of stripped-test scores
    t_b = float(np.quantile(ts.prob_promoting, 1 - r)) if r > 0 else 1.0
    rows.append({"fold": fold, "frozen": t_frozen, "clean_val_rate": r,
                 "thr_rateval": t_a, "thr_ratetest": t_b})

    # C: CDF map stripped-test scores -> clean-val score scale
    sv = np.sort(vs.prob_promoting.to_numpy())
    cv = np.sort(vc.prob_promoting.to_numpy())
    q = np.searchsorted(sv, ts.prob_promoting.to_numpy(), side="right") / len(sv)
    q = np.clip(q, 0, 1)
    mapped = np.quantile(cv, q)
    part = ts[["doc_id", "outer_fold", "stream_group", "label_promoting"]].copy()
    part["prob_mapped"] = mapped
    part["frozen_thr"] = t_frozen
    mapped_parts.append(part)

thr_df = pd.DataFrame(rows)
print(thr_df.round(3).to_string(index=False))

test = test.merge(thr_df[["fold", "thr_rateval", "thr_ratetest"]],
                  left_on="outer_fold", right_on="fold")
mapped = pd.concat(mapped_parts)

y = test.label_promoting.to_numpy(int)
results = {}
results["frozen"] = float(f1_score(y, test.pred_promoting, zero_division=0))
results["rate_match_val (label-free)"] = float(f1_score(y, (test.prob_promoting >= test.thr_rateval).astype(int), zero_division=0))
results["rate_match_test (label-free)"] = float(f1_score(y, (test.prob_promoting >= test.thr_ratetest).astype(int), zero_division=0))
ym = mapped.label_promoting.to_numpy(int)
results["cdf_map + frozen thr (label-free)"] = float(f1_score(ym, (mapped.prob_mapped >= mapped.frozen_thr).astype(int), zero_division=0))
results["labeled recalibration (reference)"] = 0.7590618336886994
results["oracle ceiling (reference)"] = 0.7837051406401552

print()
for k, v in results.items():
    print(f"{k:38s} {v:.4f}")

# paired bootstrap CI for the best label-free rule vs frozen
best_key = max(["rate_match_val (label-free)", "rate_match_test (label-free)", "cdf_map + frozen thr (label-free)"],
               key=lambda k: results[k])
if best_key.startswith("cdf"):
    pred_best = (mapped.sort_values("doc_id").prob_mapped >= mapped.sort_values("doc_id").frozen_thr).astype(int).to_numpy()
    base = mapped.sort_values("doc_id")
else:
    col = "thr_rateval" if "val" in best_key else "thr_ratetest"
    base = test.sort_values("doc_id")
    pred_best = (base.prob_promoting >= base[col]).astype(int).to_numpy()
yb = base.label_promoting.to_numpy(int)
pf = test.sort_values("doc_id").pred_promoting.to_numpy(int)
strata = [g.index.to_numpy() for _, g in base.reset_index(drop=True).groupby(["outer_fold", "stream_group"], sort=True)]
rng = np.random.default_rng(3)
boots = np.empty(5000)
for i in range(5000):
    idx = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
    boots[i] = f1_score(yb[idx], pred_best[idx], zero_division=0) - f1_score(yb[idx], pf[idx], zero_division=0)
print(f"\nbest label-free rule: {best_key}")
print(f"delta vs frozen = {results[best_key]-results['frozen']:+.4f}  95% CI [{np.quantile(boots,.025):+.4f}, {np.quantile(boots,.975):+.4f}]")

out = {"thresholds": rows, "f1": results, "best_label_free": best_key,
       "best_delta_ci": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]}
(V2 / "LABELFREE_RESULTS.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("\nsaved LABELFREE_RESULTS.json")
