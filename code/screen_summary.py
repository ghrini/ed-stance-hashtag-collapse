"""Compact screening table across aggregated systems (point estimates only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def system_row(name: str, agg_dir: Path):
    ens = pd.read_csv(agg_dir / "oof_ensemble_nested.csv")
    fs = f1_score(ens.label_suffering, ens.pred_suffering, zero_division=0)
    fp = f1_score(ens.label_promoting, ens.pred_promoting, zero_division=0)
    row = {
        "system": name,
        "macro_f1": (fs + fp) / 2,
        "suffering_f1": fs,
        "promoting_f1": fp,
        "macro_auprc": 0.5 * (average_precision_score(ens.label_suffering, ens.prob_suffering)
                              + average_precision_score(ens.label_promoting, ens.prob_promoting)),
        "brier_mean": 0.5 * (brier_score_loss(ens.label_suffering, ens.prob_suffering)
                             + brier_score_loss(ens.label_promoting, ens.prob_promoting)),
    }
    for g, x in ens.groupby("stream_group"):
        gfs = f1_score(x.label_suffering, x.pred_suffering, zero_division=0)
        gfp = f1_score(x.label_promoting, x.pred_promoting, zero_division=0)
        row[f"s{g}_macro_f1"] = (gfs + gfp) / 2
        row[f"s{g}_prom_f1"] = gfp
    nh_path = agg_dir / "oof_ensemble_nested_nohash.csv"
    if nh_path.exists():
        nh = pd.read_csv(nh_path)
        nfs = f1_score(nh.label_suffering, nh.pred_suffering, zero_division=0)
        nfp = f1_score(nh.label_promoting, nh.pred_promoting, zero_division=0)
        row["nohash_macro_f1"] = (nfs + nfp) / 2
        row["nohash_drop"] = row["macro_f1"] - row["nohash_macro_f1"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", required=True,
                    help="name=path/to/aggregate pairs")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    rows = []
    for spec in args.systems:
        name, path = spec.split("=", 1)
        rows.append(system_row(name, Path(path)))
    df = pd.DataFrame(rows).set_index("system")
    pd.set_option("display.width", 250)
    print(df.round(4).to_string())
    if args.output:
        df.to_csv(args.output)


if __name__ == "__main__":
    main()
