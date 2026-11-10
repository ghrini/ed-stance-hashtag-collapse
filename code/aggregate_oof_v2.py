"""Aggregate v2 runs into nested-threshold seed-ensemble OOF predictions.

Adapted from aggregate_oof.py: same nested-threshold logic and coverage
checks, with v2 provenance fields (encoder, learning_rate, hashtag_dropout)
and optional aggregation of the hashtag-stripped diagnostic predictions.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from splits_and_metrics import tune_threshold


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prediction_metrics(y, pred, prob):
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1),
            "auprc": float(average_precision_score(y, prob)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--fold-manifest", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    args = ap.parse_args()

    manifest = pd.read_csv(args.fold_manifest)
    expected_seeds = sorted(args.seeds)
    if manifest.doc_id.duplicated().any() or set(manifest.outer_fold) != set(range(5)):
        raise ValueError("Invalid fold manifest")
    prediction_files = sorted(glob.glob(str(Path(args.runs) / args.system / "fold_*" / "seed_*" / "predictions.csv")))
    prediction_files = [p for p in prediction_files
                        if int(Path(p).parent.name.split("_")[1]) in expected_seeds]
    if len(prediction_files) != 5 * len(expected_seeds):
        raise ValueError(f"Incomplete run file set: {len(prediction_files)}")

    root = Path(__file__).resolve().parent
    project = root.parent
    # 9a196ab0... is the pre-fp32-cast trainer; the cast is a no-op for the
    # fp32 checkpoints (bertweet-base/-large, twitter-roberta) trained with it,
    # so runs from either revision are result-equivalent.
    allowed_trainer = {sha(root / "train_fold_v2.py"),
                       "9a196ab00fb012c409253fb1f5f4c8d8341a9882a811509899e731fac7e692af"}
    expected_now = {"config_sha256": sha(args.config), "fold_manifest_sha256": sha(args.fold_manifest),
                    "model_code_sha256": sha(project / "ed_stance_model.py"),
                    "splits_code_sha256": sha(project / "splits_and_metrics.py")}
    long_parts, val_parts, nohash_parts, run_records = [], [], [], []
    for pred_file in prediction_files:
        run = json.load(open(Path(pred_file).with_name("run.json")))
        run_records.append(run)
        if run["system"] != args.system or int(run["seed"]) not in expected_seeds:
            raise ValueError("Run identity mismatch")
        for key, value in expected_now.items():
            if run.get(key) != value:
                raise ValueError(f"Stale run hash: {key} in {pred_file}")
        if run.get("trainer_code_sha256") not in allowed_trainer:
            raise ValueError(f"Unknown trainer revision in {pred_file}")
        pred = pd.read_csv(pred_file)
        long_parts.append(pred)
        val = pd.read_csv(Path(pred_file).with_name("validation_predictions.csv"))
        val["outer_fold"] = int(run["outer_fold"])
        val["model_seed"] = int(run["seed"])
        val_parts.append(val)
        nh = Path(pred_file).with_name("predictions_nohash.csv")
        if nh.exists():
            nohash_parts.append(pd.read_csv(nh))

    for key in ["encoder", "learning_rate", "hashtag_dropout", "core_sha256"]:
        if len({json.dumps(r.get(key), sort_keys=True) for r in run_records}) != 1:
            raise ValueError(f"Mixed run provenance: {key}")

    long = pd.concat(long_parts, ignore_index=True)
    vals = pd.concat(val_parts, ignore_index=True)
    if set(long.system) != {args.system} or sorted(long.model_seed.unique()) != expected_seeds:
        raise ValueError("System/seed mismatch")
    if long.duplicated(["doc_id", "model_seed"]).any():
        raise ValueError("Duplicate OOF doc/seed")
    if not np.isfinite(long[["prob_suffering", "prob_promoting"]]).all().all():
        raise ValueError("Non-finite predictions")
    if not long[["prob_suffering", "prob_promoting"]].apply(lambda c: c.between(0, 1).all()).all():
        raise ValueError("Probability outside [0,1]")
    expected_pairs = set((d, s) for d in manifest.doc_id for s in expected_seeds)
    actual_pairs = set(map(tuple, long[["doc_id", "model_seed"]].itertuples(index=False, name=None)))
    if expected_pairs != actual_pairs:
        raise ValueError("OOF document/seed coverage mismatch")
    join_cols = ["doc_id", "outer_fold", "text_hash", "cluster_id", "cluster_size",
                 "stream_group", "label_suffering", "label_promoting"]
    check = long.merge(manifest, on=join_cols, how="left", indicator=True)
    if not check._merge.eq("both").all():
        raise ValueError("OOF rows do not match frozen fold manifest")

    keys = ["doc_id", "text_hash", "cluster_id", "cluster_size", "stream_group",
            "label_suffering", "label_promoting", "system", "outer_fold"]

    def nested_ensemble(long_df, vals_df):
        ens = long_df.groupby(keys, as_index=False).agg(
            prob_suffering=("prob_suffering", "mean"), prob_promoting=("prob_promoting", "mean"))
        nested = []
        for fold in range(5):
            v = vals_df[vals_df.outer_fold.eq(fold)]
            if sorted(v.model_seed.unique()) != expected_seeds or v.duplicated(["doc_id", "model_seed"]).any():
                raise ValueError("Validation ensemble incomplete")
            vg = v.groupby(["doc_id", "label_suffering", "label_promoting"], as_index=False).agg(
                prob_suffering=("prob_suffering", "mean"), prob_promoting=("prob_promoting", "mean"))
            ts = tune_threshold(vg.label_suffering, vg.prob_suffering)
            tp = tune_threshold(vg.label_promoting, vg.prob_promoting)
            x = ens[ens.outer_fold.eq(fold)].copy()
            x["nested_threshold_suffering"] = ts
            x["nested_threshold_promoting"] = tp
            x["pred_suffering"] = x.prob_suffering.ge(ts).astype(int)
            x["pred_promoting"] = x.prob_promoting.ge(tp).astype(int)
            nested.append(x)
        return pd.concat(nested, ignore_index=True).sort_values("doc_id").reset_index(drop=True)

    ens = nested_ensemble(long, vals)
    if len(ens) != len(manifest):
        raise ValueError("Ensemble OOF count mismatch")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    long.sort_values(["doc_id", "model_seed"]).to_csv(out / "oof_long.csv", index=False)
    ens.to_csv(out / "oof_ensemble_nested.csv", index=False)
    vals.sort_values(["outer_fold", "doc_id", "model_seed"]).to_csv(out / "validation_long.csv", index=False)

    result = {"system": args.system, "encoder": run_records[0]["encoder"],
              "learning_rate": run_records[0]["learning_rate"],
              "hashtag_dropout": run_records[0]["hashtag_dropout"],
              "seeds": expected_seeds}
    nested_metrics = {"suffering": prediction_metrics(ens.label_suffering, ens.pred_suffering, ens.prob_suffering),
                      "promoting": prediction_metrics(ens.label_promoting, ens.pred_promoting, ens.prob_promoting)}
    nested_metrics["macro_f1"] = (nested_metrics["suffering"]["f1"] + nested_metrics["promoting"]["f1"]) / 2
    result["nested_development_metrics"] = nested_metrics

    if nohash_parts and len(nohash_parts) == len(prediction_files):
        nohash_long = pd.concat(nohash_parts, ignore_index=True)
        # Ensemble the diagnostic pass with the same per-fold nested thresholds.
        nh = nohash_long.groupby(keys, as_index=False).agg(
            prob_suffering=("prob_suffering", "mean"), prob_promoting=("prob_promoting", "mean"))
        thr = ens[["outer_fold", "nested_threshold_suffering", "nested_threshold_promoting"]].drop_duplicates()
        nh = nh.merge(thr, on="outer_fold", validate="many_to_one")
        nh["pred_suffering"] = nh.prob_suffering.ge(nh.nested_threshold_suffering).astype(int)
        nh["pred_promoting"] = nh.prob_promoting.ge(nh.nested_threshold_promoting).astype(int)
        nh.sort_values("doc_id").to_csv(out / "oof_ensemble_nested_nohash.csv", index=False)
        from sklearn.metrics import f1_score
        result["nohash_diagnostic"] = {
            "suffering_f1": float(f1_score(nh.label_suffering, nh.pred_suffering, zero_division=0)),
            "promoting_f1": float(f1_score(nh.label_promoting, nh.pred_promoting, zero_division=0))}

    (out / "aggregate_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
