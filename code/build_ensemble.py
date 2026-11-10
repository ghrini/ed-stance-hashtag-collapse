"""Build a nested-threshold ensemble OOF file from any set of run groups.

Each group is runs_dir:system:seed1+seed2+... ; probabilities are averaged
with equal weight across every (group, seed) member, per-fold thresholds are
re-tuned on the identically averaged validation predictions, and the output
matches the oof_ensemble_nested.csv schema used by compare_oof.py.

Examples:
  # 2-seed baseline subset for fair screening comparison
  python build_ensemble.py --group ../runs:independent:13+42 --system-name independent_s2 --output-dir ../runs_v2/independent_s2/aggregate

  # cross-architecture ensemble
  python build_ensemble.py --group ../runs:independent:13+42 --group ../runs_v2:bertweet_large:13+42 --system-name ens_bb_bl --output-dir ../runs_v2/ens_bb_bl/aggregate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from splits_and_metrics import tune_threshold
from sklearn.metrics import f1_score


def load_group(runs_dir: str, system: str, seeds: list[int]):
    preds, vals = [], []
    for fold in range(5):
        for seed in seeds:
            d = Path(runs_dir) / system / f"fold_{fold}" / f"seed_{seed}"
            p = pd.read_csv(d / "predictions.csv")
            v = pd.read_csv(d / "validation_predictions.csv")
            v["outer_fold"] = fold
            member = f"{system}/s{seed}"
            p["member"] = member
            v["member"] = member
            preds.append(p)
            vals.append(v)
    return pd.concat(preds, ignore_index=True), pd.concat(vals, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", required=True,
                    help="runs_dir:system:seed1+seed2+...")
    ap.add_argument("--system-name", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    all_preds, all_vals = [], []
    members = []
    for g in args.group:
        runs_dir, system, seed_str = g.rsplit(":", 2)
        seeds = [int(s) for s in seed_str.split("+")]
        p, v = load_group(runs_dir, system, seeds)
        members.append({"runs_dir": runs_dir, "system": system, "seeds": seeds})
        all_preds.append(p)
        all_vals.append(v)
    preds = pd.concat(all_preds, ignore_index=True)
    vals = pd.concat(all_vals, ignore_index=True)

    n_members = preds.member.nunique()
    counts = preds.groupby("doc_id").member.nunique()
    if counts.min() != n_members or counts.max() != n_members:
        raise ValueError("Every doc must be predicted by every member")

    keys = ["doc_id", "text_hash", "cluster_id", "cluster_size", "stream_group",
            "label_suffering", "label_promoting", "outer_fold"]
    ens = preds.groupby(keys, as_index=False).agg(
        prob_suffering=("prob_suffering", "mean"), prob_promoting=("prob_promoting", "mean"))
    ens["system"] = args.system_name

    nested = []
    for fold in range(5):
        v = vals[vals.outer_fold.eq(fold)]
        vc = v.groupby("doc_id").member.nunique()
        if vc.min() != n_members or vc.max() != n_members:
            raise ValueError(f"Validation coverage mismatch in fold {fold}")
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
    ens = pd.concat(nested, ignore_index=True).sort_values("doc_id").reset_index(drop=True)
    if len(ens) != 1932:
        raise ValueError(f"Expected 1932 rows, got {len(ens)}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ens.to_csv(out / "oof_ensemble_nested.csv", index=False)
    fs = f1_score(ens.label_suffering, ens.pred_suffering, zero_division=0)
    fp = f1_score(ens.label_promoting, ens.pred_promoting, zero_division=0)
    summary = {"system": args.system_name, "members": members, "n_members": int(n_members),
               "suffering_f1": float(fs), "promoting_f1": float(fp), "macro_f1": float((fs + fp) / 2)}
    (out / "ensemble_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
