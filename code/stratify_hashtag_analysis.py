"""Stratify the hashtag-dropout ablation comparison by whether a document
actually contained a hashtag.

Background: the pooled AUPRC gain for hashtag-dropout training under
hashtag-ablation (+0.040, reported in Table IV / tab:robust) looked like a
seed-consistent, threshold-free robustness result. It is not: split by
whether ablation actually changed a document's text, the gain is null in
BOTH strata. The pooled number is a between-group pooling artifact (the two
strata have different promotion prevalence and different score scales), not
a per-document improvement.

This script recomputes that finding from the canonical five-seed artifacts.
See paper/ed_stance_iemcon.tex \\S\\ref{sec:hashtag} (Section V) and
ED_STANCE_PAPER_HANDOFF/REVIEW_HISTORY.md for the audit trail that found it.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

P = Path(__file__).resolve().parent.parent


def best_f1_fast(y, p):
    o = np.argsort(-p, kind="stable")
    ys = y[o]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    tot = ys.sum()
    return float((2 * tp / (k + tot)).max()) if tot else 0.0


def had_hashtag(x):
    try:
        return len(ast.literal_eval(x)) > 0
    except Exception:
        return False


def main():
    core = pd.read_csv(P / "data" / "core_clean_deduped.csv")
    core["had_hashtag"] = core.hashtag.map(had_hashtag)
    core = core.rename(columns={"Column1": "doc_id"})[["doc_id", "had_hashtag", "stream_group"]]

    b_nh = pd.read_csv(P / "v2" / "final" / "baseline_nohash_5seed.csv").sort_values("doc_id").reset_index(drop=True)
    h_nh = pd.read_csv(P / "v2" / "final" / "htdrop_nohash_5seed.csv").sort_values("doc_id").reset_index(drop=True)

    m = (b_nh[["doc_id", "label_promoting", "prob_promoting"]]
         .rename(columns={"prob_promoting": "p_base"})
         .merge(h_nh[["doc_id", "prob_promoting"]].rename(columns={"prob_promoting": "p_htd"}), on="doc_id")
         .merge(core, on="doc_id", validate="one_to_one"))
    assert len(m) == 1932

    rng = np.random.default_rng(2026)

    def ci(sub, metric_fn, n_boot=2000):
        sub = sub.reset_index(drop=True)
        y = sub.label_promoting.to_numpy()
        pb = sub.p_base.to_numpy()
        ph = sub.p_htd.to_numpy()
        obs = metric_fn(y, ph) - metric_fn(y, pb)
        boots = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            boots[i] = metric_fn(y[idx], ph[idx]) - metric_fn(y[idx], pb[idx])
        return {"delta": float(obs), "ci_low": float(np.quantile(boots, .025)),
                "ci_high": float(np.quantile(boots, .975))}

    result = {"note": "Promotion label, hashtag-stripped test inputs. "
                      "'had_hashtag' = ablation changed this document's text; "
                      "'no_hashtag' = ablation is a no-op by construction."}
    for label, sub in [("pooled", m), ("had_hashtag", m[m.had_hashtag]), ("no_hashtag", m[~m.had_hashtag])]:
        result[label] = {
            "n": int(len(sub)), "n_pos": int(sub.label_promoting.sum()),
            "auprc": ci(sub, average_precision_score),
            "best_f1": ci(sub, best_f1_fast),
        }

    stream1_nohash = m[(~m.had_hashtag) & (m.stream_group == 1)]
    result["stream1_nohash_natural_test"] = {
        "n": int(len(stream1_nohash)), "n_pos": int(stream1_nohash.label_promoting.sum()),
        "auprc": ci(stream1_nohash, average_precision_score),
    }

    out_path = Path(__file__).with_name("STRATIFICATION_RESULTS.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
