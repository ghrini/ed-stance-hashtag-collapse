"""Does cheap recalibration (no retraining) recover the shortcut collapse?

Motivation: Section V of the paper shows that ~74% of the apparent
hashtag-ablation collapse (0.862 -> 0.542 promotion F1) is threshold
miscalibration, not lost signal -- but the "oracle" threshold used to show
that is fit directly on the outer TEST set, which is not a fair procedure.

This script asks the honest question: if you had a modest sample of
hashtag-stripped VALIDATION data (not test data), and did nothing but
re-tune the decision threshold on it, how much of the collapse would you
actually recover in practice?

Procedure per (fold, seed) of the frozen v1 independent baseline:
  1. Reconstruct the exact inner train/validation split used at training
     time (same seed formula as train_fold.py: 20260815 + outer_fold).
  2. Load the saved model_state.pt checkpoint (no retraining).
  3. Run inference on the inner-validation split with hashtags stripped.
  4. Ensemble validation probabilities across the 5 seeds within a fold.
  5. Tune a new "ablation-recalibrated" threshold on that ablated,
     seed-ensembled validation set (never touches outer-test labels).
  6. Apply that threshold to the already-computed outer-test stripped
     predictions (baseline_nohash_5seed.csv) -- a completely separate file
     from validation, so no leakage.

This is evaluated on the WHOLE test population at once (no hashtag-presence
stratification), so it is not subject to the pooling-artifact critique that
sank the hashtag-dropout AUPRC claim: there is exactly one population, one
model, one new threshold, fit honestly on held-out validation data.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ed_stance_model import EDStanceConfig, EDStanceModel
from splits_and_metrics import make_development_folds, tune_threshold
from train_fold_v2 import Collator, TweetDataset, strip_hashtags

SEEDS = [13, 42, 71, 123, 2026]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_promoting(model, df, tokenizer, batch_size=32, max_length=128):
    model.eval()
    ds = TweetDataset(df, tokenizer, max_length, False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=Collator(tokenizer))
    ids, logits = [], []
    for b in loader:
        ids.extend(b.pop("doc_id").tolist())
        inputs = {k: v.to(DEVICE) for k, v in b.items()}
        out = model(**inputs)
        logits.extend(out["promotion_logits"].float().cpu().tolist())
    prob = 1 / (1 + np.exp(-np.array(logits)))
    return pd.DataFrame({"doc_id": ids, "prob_promoting": prob})


def main():
    core = pd.read_csv(P / "data" / "core_clean_deduped.csv")
    manifest = pd.read_csv(P / "artifacts" / "dev_outer_folds_v1.csv")
    core = core.merge(manifest[["doc_id", "outer_fold", "text_hash", "cluster_id", "cluster_size"]],
                      left_on="Column1", right_on="doc_id", how="left", validate="one_to_one")

    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True, use_fast=False)
    config = EDStanceConfig(encoder_name="vinai/bertweet-base", dropout=0.15,
                            use_dependency_head=False, use_hierarchy_penalty=False)

    test_strip = pd.read_csv(P / "v2" / "final" / "baseline_nohash_5seed.csv")

    recal_thresholds = {}
    per_seed_val_preds = {f: [] for f in range(5)}

    for fold in range(5):
        outer_train = core[core.outer_fold.ne(fold)].copy()
        inner = make_development_folds(outer_train, 5, 20260815 + fold)
        val = outer_train[inner == 0].copy()
        val_strip = val.copy()
        val_strip["text_orig"] = val_strip["text_orig"].map(strip_hashtags)

        for seed in SEEDS:
            ckpt = P / "runs" / "independent" / f"fold_{fold}" / f"seed_{seed}" / "model_state.pt"
            model = EDStanceModel(config).to(DEVICE)
            state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state, strict=True)
            pred = predict_promoting(model, val_strip, tokenizer)
            pred = pred.merge(val[["Column1", "label_promoting"]].rename(columns={"Column1": "doc_id"}),
                              on="doc_id", validate="one_to_one")
            per_seed_val_preds[fold].append(pred)
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            print(f"fold={fold} seed={seed} done ({len(pred)} val docs)", flush=True)

        vg = pd.concat(per_seed_val_preds[fold]).groupby(
            ["doc_id", "label_promoting"], as_index=False)["prob_promoting"].mean()
        t = tune_threshold(vg.label_promoting, vg.prob_promoting)
        recal_thresholds[fold] = t
        print(f"fold={fold} recalibrated (ablated-validation) threshold = {t:.3f}", flush=True)

    test_strip["recal_threshold"] = test_strip.outer_fold.map(recal_thresholds)
    test_strip["pred_recal"] = (test_strip.prob_promoting >= test_strip.recal_threshold).astype(int)

    y = test_strip.label_promoting.to_numpy(int)
    f1_frozen = f1_score(y, test_strip.pred_promoting, zero_division=0)
    f1_recal = f1_score(y, test_strip.pred_recal, zero_division=0)

    rng = np.random.default_rng(20260818)
    strata = [g.index.to_numpy() for _, g in test_strip.reset_index(drop=True).groupby(
        ["outer_fold", "stream_group"], sort=True)]
    ts = test_strip.reset_index(drop=True)
    yv = ts.label_promoting.to_numpy(int)
    pf = ts.pred_promoting.to_numpy(int)
    pr = ts.pred_recal.to_numpy(int)
    obs = f1_score(yv, pr, zero_division=0) - f1_score(yv, pf, zero_division=0)
    boots = np.empty(10000)
    for i in range(10000):
        idx = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        boots[i] = f1_score(yv[idx], pr[idx], zero_division=0) - f1_score(yv[idx], pf[idx], zero_division=0)

    result = {
        "note": "Threshold re-tuned on hashtag-stripped INNER-VALIDATION predictions (honest, "
                "no test labels touched), then applied to the outer-test stripped predictions.",
        "recalibrated_thresholds_by_fold": recal_thresholds,
        "f1_frozen_clean_threshold": float(f1_frozen),
        "f1_recalibrated": float(f1_recal),
        "f1_oracle_upper_bound": 0.7837051406401552,
        "f1_clean_best_case": 0.8684210526315789,
        "delta_recal_minus_frozen": {"delta": float(obs), "ci_low": float(np.quantile(boots, .025)),
                                     "ci_high": float(np.quantile(boots, .975))},
        "fraction_of_oracle_gap_recovered": float((f1_recal - f1_frozen) / (0.7837051406401552 - f1_frozen)),
    }
    out_path = Path(__file__).with_name("RECALIBRATION_RESULTS.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
