"""Hashtag-stripped diagnostic predictions for the frozen baseline.

Loads each saved independent-run checkpoint and predicts on its outer-fold
test items with all hashtags removed, mirroring the v2 trainers' built-in
diagnostic. CPU-friendly (inference only) so it can run beside GPU training.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ed_stance_model import EDStanceConfig, EDStanceModel
from train_fold_v2 import Collator, TweetDataset, strip_hashtags
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


@torch.no_grad()
def predict(model, df, tokenizer, device, max_length=128, batch_size=32):
    model.eval()
    ds = TweetDataset(df, tokenizer, max_length, False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=Collator(tokenizer))
    ids, ls, lp = [], [], []
    for b in loader:
        ids.extend(b.pop("doc_id").tolist())
        inputs = {k: v.to(device) for k, v in b.items()}
        out = model(**inputs)
        ls.extend(out["suffering_logits"].float().cpu().tolist())
        lp.extend(out["promotion_logits"].float().cpu().tolist())
    import numpy as np
    return pd.DataFrame({"doc_id": ids, "logit_suffering": ls, "logit_promoting": lp}).assign(
        prob_suffering=lambda x: 1 / (1 + np.exp(-x.logit_suffering)),
        prob_promoting=lambda x: 1 / (1 + np.exp(-x.logit_promoting)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(Path(__file__).resolve().parent.parent / "runs"))
    ap.add_argument("--system", default="independent")
    ap.add_argument("--core", default=str(Path(__file__).resolve().parent.parent / "data" / "core_clean_deduped.csv"))
    ap.add_argument("--fold-manifest", default=str(Path(__file__).resolve().parent.parent / "artifacts" / "dev_outer_folds_v1.csv"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 71, 123, 2026])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True, use_fast=False)
    core = pd.read_csv(args.core)
    manifest = pd.read_csv(args.fold_manifest)
    core = core.merge(manifest[["doc_id", "outer_fold", "text_hash", "cluster_id", "cluster_size"]],
                      left_on="Column1", right_on="doc_id", how="left", validate="one_to_one")

    config = EDStanceConfig(encoder_name="vinai/bertweet-base", dropout=0.15,
                            use_dependency_head=False, use_hierarchy_penalty=False)
    for fold in range(5):
        test = core[core.outer_fold.eq(fold)].copy()
        test["text_orig"] = test["text_orig"].map(strip_hashtags)
        key_cols = test[["Column1", "text_hash", "cluster_id", "cluster_size", "stream_group",
                         "label_suffering", "label_promoting"]].rename(columns={"Column1": "doc_id"})
        for seed in args.seeds:
            d = Path(args.runs) / args.system / f"fold_{fold}" / f"seed_{seed}"
            out_file = d / "predictions_nohash.csv"
            if out_file.exists():
                print(f"SKIP fold={fold} seed={seed}", flush=True)
                continue
            state = torch.load(d / "model_state.pt", map_location=device, weights_only=True)
            model = EDStanceModel(config).to(device)
            missing, unexpected = model.load_state_dict(state, strict=True), None
            run = json.load(open(d / "run.json"))
            ts = run["thresholds"]["suffering"]
            tp = run["thresholds"]["promoting"]
            pred = key_cols.merge(predict(model, test, tokenizer, device), on="doc_id", validate="one_to_one")
            pred["pred_suffering"] = pred.prob_suffering.ge(ts).astype(int)
            pred["pred_promoting"] = pred.prob_promoting.ge(tp).astype(int)
            pred["outer_fold"] = fold
            pred["model_seed"] = seed
            pred["system"] = args.system
            pred.to_csv(out_file, index=False)
            print(f"DONE fold={fold} seed={seed}", flush=True)
            del model


if __name__ == "__main__":
    main()
