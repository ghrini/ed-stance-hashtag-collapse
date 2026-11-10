"""Save hashtag-stripped inner-validation predictions for the v1 baseline.

Same inference as recalibration_recovery.py but persists the per-fold
seed-ensembled validation predictions (clean AND stripped) to CSV so that
label-free threshold rules (rate matching, quantile mapping) can be
evaluated offline without re-running the GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ed_stance_model import EDStanceConfig, EDStanceModel
from splits_and_metrics import make_development_folds
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
    core = core.merge(manifest[["doc_id", "outer_fold"]],
                      left_on="Column1", right_on="doc_id", how="left", validate="one_to_one")

    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True, use_fast=False)
    config = EDStanceConfig(encoder_name="vinai/bertweet-base", dropout=0.15,
                            use_dependency_head=False, use_hierarchy_penalty=False)

    rows = []
    for fold in range(5):
        outer_train = core[core.outer_fold.ne(fold)].copy()
        inner = make_development_folds(outer_train, 5, 20260815 + fold)
        val = outer_train[inner == 0].copy()
        val_strip = val.copy()
        val_strip["text_orig"] = val_strip["text_orig"].map(strip_hashtags)

        per_seed_strip, per_seed_clean = [], []
        for seed in SEEDS:
            ckpt = P / "runs" / "independent" / f"fold_{fold}" / f"seed_{seed}" / "model_state.pt"
            model = EDStanceModel(config).to(DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True), strict=True)
            per_seed_strip.append(predict_promoting(model, val_strip, tokenizer).assign(seed=seed))
            per_seed_clean.append(predict_promoting(model, val, tokenizer).assign(seed=seed))
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            print(f"fold={fold} seed={seed} done", flush=True)

        for tag, parts in [("stripped", per_seed_strip), ("clean", per_seed_clean)]:
            g = pd.concat(parts).groupby("doc_id", as_index=False)["prob_promoting"].mean()
            g = g.merge(val[["Column1", "label_promoting"]].rename(columns={"Column1": "doc_id"}),
                        on="doc_id", validate="one_to_one")
            g["outer_fold"] = fold
            g["view"] = tag
            rows.append(g)

    out = pd.concat(rows, ignore_index=True)
    out_path = Path(__file__).with_name("baseline_val_preds_clean_and_stripped.csv")
    out.to_csv(out_path, index=False)
    print("saved", out_path, len(out), "rows")


if __name__ == "__main__":
    main()
