"""Does the threshold repair still work with GENUINELY held-out validation
data -- data the model calibrating on it has never been trained on at all?

Why this exists: recalibration_recovery.py loads `model_state.pt`, the
FINAL per-fold checkpoint -- but that checkpoint is trained on the FULL
outer-training set, which *includes* the inner-validation documents
(confirmed in train_fold.py: the outer-stage model is fit on `outer_train`,
and `val = outer_train[inner==0]` is a subset of it). So the "validation"
predictions used to fit the ablated threshold in recalibration_recovery.py
come from a model that has already seen those exact documents (clean text,
gold labels) during its own training -- just never with hashtags stripped.
That is milder than genuine held-out calibration data and could optimistically
bias the recovery estimate.

This script tests the strict version: retrain the INNER-stage model (fit on
inner_train ONLY, exactly as train_fold.py's own inner stage does, then
discarded there) for each fold and seed, so `val` is truly unseen by it.
Use its predictions on hashtag-stripped `val` (now genuinely fresh) to fit
a threshold, and apply that threshold to this SAME inner-only model's own
predictions on the outer-test fold (also never seen by it). Report both
this model's own clean/ablated collapse and how much of it the fresh-data
threshold recovers, for direct comparison against the paper's headline
numbers (which use the full/final model).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score

P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ed_stance_model import EDStanceConfig, EDStanceModel  # noqa: F401 (used indirectly via train_fold)
from splits_and_metrics import make_development_folds, tune_threshold
from train_fold import fit, make_model, predict, seed_all, SYSTEMS
from train_fold_v2 import strip_hashtags
from transformers import AutoTokenizer

SEEDS = [13, 42, 71, 123, 2026]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def f1_at_threshold(y, p, t):
    return float(f1_score(y, (p >= t).astype(int), zero_division=0))


def best_f1(y, p):
    order = np.argsort(-p, kind="stable")
    ys = y[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    tot = ys.sum()
    return float((2 * tp / (k + tot)).max()) if tot else 0.0


def main():
    cfg = json.load(open(P / "config.json"))
    core = pd.read_csv(P / "data" / "core_clean_deduped.csv")
    manifest = pd.read_csv(P / "artifacts" / "dev_outer_folds_v1.csv")
    core = core.merge(manifest[["doc_id", "outer_fold", "text_hash", "cluster_id", "cluster_size"]],
                      left_on="Column1", right_on="doc_id", how="left", validate="one_to_one")
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True, use_fast=False)
    spec = SYSTEMS["independent"]

    per_fold_seed = []
    t0 = time.time()
    for fold in range(5):
        outer_train = core[core.outer_fold.ne(fold)].copy()
        test = core[core.outer_fold.eq(fold)].copy()
        inner = make_development_folds(outer_train, 5, 20260815 + fold)
        inner_train = outer_train[inner != 0].copy()
        val = outer_train[inner == 0].copy()
        val_strip = val.copy()
        val_strip["text_orig"] = val_strip["text_orig"].map(strip_hashtags)
        test_strip = test.copy()
        test_strip["text_orig"] = test_strip["text_orig"].map(strip_hashtags)

        for seed in SEEDS:
            seed_all(seed)
            model = make_model("vinai/bertweet-base", spec, cfg).to(DEVICE)
            fit(model, inner_train, None, tokenizer, cfg, seed, DEVICE)

            val_clean_pred = predict(model, val, tokenizer, cfg, DEVICE)
            val_strip_pred = predict(model, val_strip, tokenizer, cfg, DEVICE)
            test_clean_pred = predict(model, test, tokenizer, cfg, DEVICE)
            test_strip_pred = predict(model, test_strip, tokenizer, cfg, DEVICE)
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

            yv = val.label_promoting.to_numpy(int)
            clean_thr = tune_threshold(val.label_promoting, val_clean_pred.prob_promoting.to_numpy())
            fresh_ablated_thr = tune_threshold(val.label_promoting, val_strip_pred.prob_promoting.to_numpy())

            yt = test.label_promoting.to_numpy(int)
            p_test_clean = test_clean_pred.prob_promoting.to_numpy()
            p_test_strip = test_strip_pred.prob_promoting.to_numpy()

            row = {
                "fold": fold, "seed": seed,
                "clean_val_threshold": float(clean_thr),
                "fresh_ablated_val_threshold": float(fresh_ablated_thr),
                "test_clean_f1_at_clean_thr": f1_at_threshold(yt, p_test_clean, clean_thr),
                "test_stripped_f1_at_FROZEN_clean_thr": f1_at_threshold(yt, p_test_strip, clean_thr),
                "test_stripped_f1_at_FRESH_recalibrated_thr": f1_at_threshold(yt, p_test_strip, fresh_ablated_thr),
                "test_stripped_best_f1_oracle": best_f1(yt, p_test_strip),
                "test_stripped_auprc": float(average_precision_score(yt, p_test_strip)),
                "test_clean_auprc": float(average_precision_score(yt, p_test_clean)),
            }
            per_fold_seed.append(row)
            elapsed = time.time() - t0
            print(f"fold={fold} seed={seed} done  frozen={row['test_stripped_f1_at_FROZEN_clean_thr']:.4f} "
                  f"fresh_recal={row['test_stripped_f1_at_FRESH_recalibrated_thr']:.4f}  "
                  f"elapsed={elapsed:.0f}s", flush=True)

    df = pd.DataFrame(per_fold_seed)
    df.to_csv(Path(__file__).with_name("genuine_heldout_perfoldseed.csv"), index=False)

    # Seed-ensembled per fold, then pooled, mirroring the rest of the paper's method.
    ens = df.groupby("fold").mean(numeric_only=True)
    summary = {
        "note": "Inner-only model (never trained on its own validation split), "
                "genuinely fresh ablated-validation recalibration, evaluated on "
                "outer-test. Seed-averaged per fold, then simple mean across folds "
                "(unweighted, since fold sizes are ~equal by construction).",
        "n_folds": 5, "n_seeds": len(SEEDS),
        "mean_test_clean_f1_at_clean_thr": float(ens.test_clean_f1_at_clean_thr.mean()),
        "mean_test_stripped_f1_at_FROZEN_clean_thr": float(ens.test_stripped_f1_at_FROZEN_clean_thr.mean()),
        "mean_test_stripped_f1_at_FRESH_recalibrated_thr": float(ens.test_stripped_f1_at_FRESH_recalibrated_thr.mean()),
        "mean_test_stripped_best_f1_oracle": float(ens.test_stripped_best_f1_oracle.mean()),
        "delta_fresh_recal_minus_frozen": float(ens.test_stripped_f1_at_FRESH_recalibrated_thr.mean()
                                                - ens.test_stripped_f1_at_FROZEN_clean_thr.mean()),
        "fraction_of_own_oracle_gap_recovered": None,
    }
    gap = summary["mean_test_stripped_best_f1_oracle"] - summary["mean_test_stripped_f1_at_FROZEN_clean_thr"]
    if gap > 0:
        summary["fraction_of_own_oracle_gap_recovered"] = float(
            summary["delta_fresh_recal_minus_frozen"] / gap)

    out = {"summary": summary, "per_fold_seed": per_fold_seed}
    out_path = Path(__file__).with_name("GENUINE_HELDOUT_RESULTS.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
