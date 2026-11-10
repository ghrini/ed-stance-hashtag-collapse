"""Resumable v2 screening orchestrator: encoders + hashtag dropout.

Runs each (system, fold, seed) as an independent subprocess, skipping
completed runs, then aggregates every finished system.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

SPECS = {
    "bertweet_large":        {"encoder": "vinai/bertweet-large",                        "lr": 1e-5, "htdrop": 0.0},
    "troberta_2022":         {"encoder": "cardiffnlp/twitter-roberta-base-2022-154m",   "lr": 2e-5, "htdrop": 0.0},
    "deberta_v3":            {"encoder": "microsoft/deberta-v3-base",                   "lr": 2e-5, "htdrop": 0.0},
    "bertweet_htdrop":       {"encoder": "vinai/bertweet-base",                          "lr": 2e-5, "htdrop": 0.5},
    "bertweet_large_htdrop": {"encoder": "vinai/bertweet-large",                        "lr": 1e-5, "htdrop": 0.5},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", required=True, choices=sorted(SPECS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[13, 42])
    ap.add_argument("--core", default=str(PROJECT / "data" / "core_clean_deduped.csv"))
    ap.add_argument("--fold-manifest", default=str(PROJECT / "artifacts" / "dev_outer_folds_v1.csv"))
    ap.add_argument("--config", default=str(HERE / "config_v2.json"))
    ap.add_argument("--output-dir", default=str(PROJECT / "runs_v2"))
    args = ap.parse_args()

    trainer = HERE / "train_fold_v2.py"
    started = time.time()
    for system in args.systems:
        spec = SPECS[system]
        for fold in range(5):
            for seed in args.seeds:
                out = Path(args.output_dir) / system / f"fold_{fold}" / f"seed_{seed}"
                required = [out / "run.json", out / "predictions.csv",
                            out / "validation_predictions.csv", out / "predictions_nohash.csv"]
                if all(p.exists() for p in required):
                    run = json.load(open(out / "run.json"))
                    if (run.get("system") == system and int(run.get("outer_fold", -1)) == fold
                            and int(run.get("seed", -1)) == seed and run.get("encoder") == spec["encoder"]):
                        print(f"SKIP {system} fold={fold} seed={seed}", flush=True)
                        continue
                    raise ValueError(f"Stale run directory: {out}")
                if any(p.exists() for p in required):
                    raise ValueError(f"Partial run directory must be moved aside: {out}")
                cmd = [sys.executable, str(trainer),
                       "--core", args.core, "--fold-manifest", args.fold_manifest,
                       "--config", args.config, "--outer-fold", str(fold),
                       "--system", system, "--seed", str(seed),
                       "--encoder", spec["encoder"], "--learning-rate", str(spec["lr"]),
                       "--hashtag-dropout", str(spec["htdrop"]),
                       "--output-dir", str(out)]
                print(f"RUN {system} fold={fold} seed={seed} elapsed={time.time()-started:.0f}s", flush=True)
                subprocess.run(cmd, check=True)
    for system in args.systems:
        subprocess.run([sys.executable, str(HERE / "aggregate_oof_v2.py"),
                        "--runs", args.output_dir, "--system", system,
                        "--output-dir", str(Path(args.output_dir) / system / "aggregate"),
                        "--fold-manifest", args.fold_manifest, "--config", args.config,
                        "--seeds", *[str(x) for x in args.seeds]], check=True)
    print(f"ALL DONE elapsed={time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
