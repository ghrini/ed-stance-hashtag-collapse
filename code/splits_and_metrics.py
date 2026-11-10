"""Deterministic development splits, thresholding, and item-level metrics."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_recall_fscore_support


def normalize_for_grouping(text: str) -> str:
    text = str(text).casefold()
    text = re.sub(r"https?://\S+|www\.\S+", "<url>", text)
    text = re.sub(r"@\w+", "<user>", text)
    return re.sub(r"\s+", " ", text).strip()


def stable_text_id(text: str) -> str:
    return hashlib.sha256(normalize_for_grouping(text).encode("utf-8")).hexdigest()


def make_development_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 20260815) -> np.ndarray:
    """Greedily balance both labels, rare joint states, and collection stream.

    The retained dataset has one representative per near-duplicate cluster.
    The deterministic objective below produces folds of 386--387 rows with
    promotion 91--92, suffering 190--191, and promotion-only 2--3 per fold.
    """
    s = df["label_suffering"].to_numpy(int)
    p = df["label_promoting"].to_numpy(int)
    streams = [df["stream_group"].eq(g).to_numpy(int) for g in (1, 2, 3)]
    y = np.column_stack([s, p, s & p, (1 - s) & p, *streams])
    rng = np.random.default_rng(seed)
    rarity = (y / np.maximum(y.sum(axis=0), 1)).sum(axis=1)
    order = np.lexsort((rng.random(len(df)), -rarity))
    target = y.sum(axis=0) / n_splits
    fold_counts = np.zeros((n_splits, y.shape[1]), dtype=int)
    sizes = np.zeros(n_splits, dtype=int)
    folds = np.full(len(df), -1, dtype=int)
    for i in order:
        objectives = []
        for fold in range(n_splits):
            counts = fold_counts.copy(); counts[fold] += y[i]
            proposed_sizes = sizes.copy(); proposed_sizes[fold] += 1
            label_error = ((counts - target) ** 2 / np.maximum(target, 1)).sum()
            size_error = ((proposed_sizes - len(df) / n_splits) ** 2 / (len(df) / n_splits)).sum()
            objectives.append(label_error + 0.2 * size_error)
        best = min(range(n_splits), key=lambda f: (objectives[f], sizes[f], f))
        folds[i] = best; fold_counts[best] += y[i]; sizes[best] += 1
    assert np.all(folds >= 0)
    return folds


def tune_threshold(y_true, probability, grid=None):
    grid = np.asarray(grid if grid is not None else np.linspace(0.05, 0.95, 91))
    scores = np.array([f1_score(y_true, probability >= t, zero_division=0) for t in grid])
    best = scores.max()
    # Stable tie rule: threshold closest to 0.5, then higher threshold.
    candidates = grid[np.isclose(scores, best)]
    return float(sorted(candidates, key=lambda t: (abs(t - 0.5), -t))[0])


def label_metrics(y_true, probability, threshold):
    pred = probability >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "auprc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)), "threshold": float(threshold),
    }


def paired_bootstrap_delta(y_s, y_p, base_s, base_p, new_s, new_p, thresholds, n_boot=10000, seed=2026):
    rng = np.random.default_rng(seed)
    n = len(y_s)
    def score(ps, pp, idx, prefix):
        fs = f1_score(y_s[idx], ps[idx] >= thresholds[f"{prefix}_s"], zero_division=0)
        fp = f1_score(y_p[idx], pp[idx] >= thresholds[f"{prefix}_p"], zero_division=0)
        return (fs + fp) / 2
    observed = score(new_s, new_p, np.arange(n), "new") - score(base_s, base_p, np.arange(n), "base")
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[b] = score(new_s, new_p, idx, "new") - score(base_s, base_p, idx, "base")
    return {"delta": float(observed), "ci_low": float(np.quantile(deltas, .025)), "ci_high": float(np.quantile(deltas, .975))}
