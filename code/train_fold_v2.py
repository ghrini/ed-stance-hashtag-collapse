"""ED-STANCE v2: train one leakage-controlled outer fold with a configurable
encoder and optional hashtag-dropout augmentation; emit item-level predictions.

Identical protocol to train_fold.py (same folds, schedule, inner-split
threshold tuning, four-exposure training) so v2 OOF predictions pair
item-by-item with the frozen baseline. Differences:
  - independent two-head model only (no dependency head, no contrastive)
  - encoder is a required CLI choice with per-encoder learning rate
  - optional per-hashtag stochastic dropout during training
  - an extra prediction pass on hashtag-stripped test text for the
    shortcut-robustness diagnostic
  - fold model weights are not saved (final models are retrained separately)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ed_stance_model import EDStanceConfig, EDStanceModel
from splits_and_metrics import make_development_folds, tune_threshold

HASHTAG_RE = re.compile(r"#\w+")


def strip_hashtags(text: str) -> str:
    return re.sub(r"\s+", " ", HASHTAG_RE.sub("", str(text))).strip()


def load_tokenizer(encoder: str):
    if "bertweet" in encoder.lower():
        return AutoTokenizer.from_pretrained(encoder, normalization=True, use_fast=False)
    try:
        return AutoTokenizer.from_pretrained(encoder)
    except Exception:
        return AutoTokenizer.from_pretrained(encoder, use_fast=False)


class TweetDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128, labels=True, hashtag_dropout=0.0, rng=None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels
        self.hashtag_dropout = float(hashtag_dropout)
        self.rng = rng

    def __len__(self):
        return len(self.df)

    def _augment(self, text: str) -> str:
        if self.hashtag_dropout <= 0 or self.rng is None:
            return text
        out = HASHTAG_RE.sub(
            lambda m: "" if self.rng.random() < self.hashtag_dropout else m.group(0), text)
        return re.sub(r"\s+", " ", out).strip() or text

    def __getitem__(self, i):
        r = self.df.iloc[i]
        text = self._augment(str(r.text_orig)) if self.labels else str(r.text_orig)
        item = self.tokenizer(text, truncation=True, max_length=self.max_length)
        item["doc_id"] = int(r.Column1)
        if self.labels:
            item["suffering_labels"] = float(r.label_suffering)
            item["promotion_labels"] = float(r.label_promoting)
        return item


class Collator:
    def __init__(self, tokenizer):
        self.base = DataCollatorWithPadding(tokenizer, return_tensors="pt")

    def __call__(self, batch):
        doc = torch.tensor([x.pop("doc_id") for x in batch], dtype=torch.long)
        ys = torch.tensor([x.pop("suffering_labels") for x in batch], dtype=torch.float) if "suffering_labels" in batch[0] else None
        yp = torch.tensor([x.pop("promotion_labels") for x in batch], dtype=torch.float) if "promotion_labels" in batch[0] else None
        out = self.base(batch)
        out["doc_id"] = doc
        if ys is not None:
            out["suffering_labels"] = ys
            out["promotion_labels"] = yp
        return out


def seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def make_model(encoder, cfg):
    config = EDStanceConfig(encoder_name=encoder, dropout=cfg["dropout"],
                            use_dependency_head=False, use_hierarchy_penalty=False)
    # Some checkpoints (e.g. deberta-v3-base) ship fp16 weights; AMP GradScaler
    # requires fp32 master weights, so force full precision after loading.
    return EDStanceModel(config).float()


def optimizer_for(model, encoder_lr, cfg):
    head, encoder = [], []
    for name, p in model.named_parameters():
        (encoder if name.startswith("encoder.") else head).append(p)
    return torch.optim.AdamW(
        [{"params": encoder, "lr": encoder_lr}, {"params": head, "lr": 1e-4}],
        weight_decay=cfg["weight_decay"])


def fit(model, train_df, tokenizer, cfg, seed, device, encoder_lr, hashtag_dropout):
    generator = torch.Generator()
    generator.manual_seed(seed)
    aug_rng = np.random.default_rng(seed * 100003 + 7) if hashtag_dropout > 0 else None
    ds = TweetDataset(train_df, tokenizer, cfg["max_length"], True, hashtag_dropout, aug_rng)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, generator=generator,
                        collate_fn=Collator(tokenizer))
    opt = optimizer_for(model, encoder_lr, cfg)
    epochs = int(cfg["epochs"])
    accum = int(cfg["gradient_accumulation_steps"])
    total_microbatches = len(loader) * epochs
    total_updates = int(np.ceil(total_microbatches / accum))
    sched = get_linear_schedule_with_warmup(opt, int(total_updates * cfg["warmup_ratio"]), total_updates)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model.train()
    opt.zero_grad(set_to_none=True)
    exposure = []
    nonpadding = 0
    steps = 0
    start = time.time()
    global_micro = 0
    for epoch in range(epochs):
        for batch in loader:
            global_micro += 1
            exposure.extend(batch["doc_id"].tolist())
            nonpadding += int(batch["attention_mask"].sum())
            labels = {"suffering_labels": batch["suffering_labels"].to(device),
                      "promotion_labels": batch["promotion_labels"].to(device)}
            inputs = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(**inputs, **labels)
                loss = out["loss"]
                remainder = total_microbatches % accum
                window_size = remainder if remainder and global_micro > total_microbatches - remainder else accum
                loss = loss / window_size
            scaler.scale(loss).backward()
            if global_micro % accum == 0 or global_micro == total_microbatches:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                steps += 1
    counts = pd.Series(exposure).value_counts()
    assert len(counts) == len(train_df) and counts.min() == epochs and counts.max() == epochs
    assert steps == total_updates
    return {"epochs": epochs, "gold_rows": len(train_df), "gold_exposures": len(exposure),
            "optimizer_steps": steps, "nonpadding_tokens": nonpadding,
            "hashtag_dropout": hashtag_dropout, "encoder_lr": encoder_lr,
            "wall_seconds": time.time() - start}


@torch.no_grad()
def predict(model, df, tokenizer, cfg, device):
    model.eval()
    ds = TweetDataset(df, tokenizer, cfg["max_length"], False)
    loader = DataLoader(ds, batch_size=cfg["batch_size"] * 2, shuffle=False, collate_fn=Collator(tokenizer))
    ids, ls, lp = [], [], []
    for b in loader:
        ids.extend(b.pop("doc_id").tolist())
        inputs = {k: v.to(device) for k, v in b.items()}
        out = model(**inputs)
        ls.extend(out["suffering_logits"].float().cpu().tolist())
        lp.extend(out["promotion_logits"].float().cpu().tolist())
    return pd.DataFrame({"doc_id": ids, "logit_suffering": ls, "logit_promoting": lp}).assign(
        prob_suffering=lambda x: 1 / (1 + np.exp(-x.logit_suffering)),
        prob_promoting=lambda x: 1 / (1 + np.exp(-x.logit_promoting)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", required=True)
    ap.add_argument("--fold-manifest", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--outer-fold", type=int, required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--learning-rate", type=float, required=True)
    ap.add_argument("--hashtag-dropout", type=float, default=0.0)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args.encoder)
    core = pd.read_csv(args.core)
    manifest = pd.read_csv(args.fold_manifest)
    if manifest.doc_id.duplicated().any() or core.Column1.duplicated().any():
        raise ValueError("Duplicate document IDs")
    if set(manifest.outer_fold) != set(range(5)) or set(core.Column1) != set(manifest.doc_id):
        raise ValueError("Core/fold manifest mismatch")
    core = core.merge(manifest[["doc_id", "outer_fold", "text_hash", "cluster_id", "cluster_size"]],
                      left_on="Column1", right_on="doc_id", how="left", validate="one_to_one")
    if core.outer_fold.isna().any():
        raise ValueError("Unassigned core row")
    outer_train = core[core.outer_fold.ne(args.outer_fold)].copy()
    test = core[core.outer_fold.eq(args.outer_fold)].copy()
    if set(outer_train.Column1) & set(test.Column1) or len(outer_train) + len(test) != len(core):
        raise ValueError("Outer split overlap/incompleteness")

    inner = make_development_folds(outer_train, 5, 20260815 + args.outer_fold)
    inner_train = outer_train[inner != 0].copy()
    val = outer_train[inner == 0].copy()

    model = make_model(args.encoder, cfg).to(device)
    inner_ledger = fit(model, inner_train, tokenizer, cfg, args.seed, device,
                       args.learning_rate, args.hashtag_dropout)
    val_pred = predict(model, val, tokenizer, cfg, device)
    val_pred = val[["Column1", "label_suffering", "label_promoting"]].rename(
        columns={"Column1": "doc_id"}).merge(val_pred, on="doc_id", validate="one_to_one")
    ts = tune_threshold(val_pred.label_suffering, val_pred.prob_suffering)
    tp = tune_threshold(val_pred.label_promoting, val_pred.prob_promoting)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    seed_all(args.seed)
    model = make_model(args.encoder, cfg).to(device)
    outer_ledger = fit(model, outer_train, tokenizer, cfg, args.seed, device,
                       args.learning_rate, args.hashtag_dropout)
    pred = predict(model, test, tokenizer, cfg, device)
    key_cols = test[["Column1", "text_hash", "cluster_id", "cluster_size", "stream_group",
                     "label_suffering", "label_promoting"]].rename(columns={"Column1": "doc_id"})
    pred = key_cols.merge(pred, on="doc_id", validate="one_to_one")
    pred["pred_suffering"] = pred.prob_suffering.ge(ts).astype(int)
    pred["pred_promoting"] = pred.prob_promoting.ge(tp).astype(int)
    pred["outer_fold"] = args.outer_fold
    pred["model_seed"] = args.seed
    pred["system"] = args.system
    pred["threshold_suffering"] = ts
    pred["threshold_promoting"] = tp

    # Shortcut diagnostic: same model, all hashtags stripped from test text.
    test_nohash = test.copy()
    test_nohash["text_orig"] = test_nohash["text_orig"].map(strip_hashtags)
    pred_nohash = key_cols.merge(predict(model, test_nohash, tokenizer, cfg, device),
                                 on="doc_id", validate="one_to_one")
    pred_nohash["pred_suffering"] = pred_nohash.prob_suffering.ge(ts).astype(int)
    pred_nohash["pred_promoting"] = pred_nohash.prob_promoting.ge(tp).astype(int)
    pred_nohash["outer_fold"] = args.outer_fold
    pred_nohash["model_seed"] = args.seed
    pred_nohash["system"] = args.system

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out / "predictions.csv", index=False)
    val_pred.to_csv(out / "validation_predictions.csv", index=False)
    pred_nohash.to_csv(out / "predictions_nohash.csv", index=False)

    root = Path(__file__).resolve().parent.parent
    run = {"system": args.system, "outer_fold": args.outer_fold, "seed": args.seed,
           "encoder": args.encoder, "encoder_revision": getattr(model.encoder.config, "_commit_hash", None),
           "learning_rate": args.learning_rate, "hashtag_dropout": args.hashtag_dropout,
           "device": str(device), "thresholds": {"suffering": ts, "promoting": tp},
           "inner_ledger": inner_ledger, "outer_ledger": outer_ledger,
           "core_sha256": sha(args.core), "fold_manifest_sha256": sha(args.fold_manifest),
           "config_sha256": sha(args.config),
           "model_code_sha256": sha(root / "ed_stance_model.py"),
           "trainer_code_sha256": sha(__file__),
           "splits_code_sha256": sha(root / "splits_and_metrics.py"),
           "test_rows": len(test)}
    (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps({k: run[k] for k in ("system", "outer_fold", "seed", "encoder", "thresholds")}, indent=2))


if __name__ == "__main__":
    main()
