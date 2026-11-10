"""Generate the full figure set for the ED-STANCE paper and analysis.

Writes PDF (for LaTeX) and PNG (for viewing) into paper/figures/.
Every figure is generated from the canonical artifacts listed in
runs_v2/READ_THIS_BEFORE_USING_AGGREGATES.md - no aggregate/ directories.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve)

P = Path(r"D:\ED_STANCE_Windows_Auto_Setup\ed_stance")
OUT = Path(r"D:\ED_STANCE_Windows_Auto_Setup\paper\figures")
OUT.mkdir(parents=True, exist_ok=True)

BASE, HTD, ENS = "#4878CF", "#D65F5F", "#6ACC64"
MID, FROZ = "#6ACC64", "#EE854A"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})

b_cl = pd.read_csv(P / "runs/independent/aggregate/oof_ensemble_nested.csv").sort_values("doc_id")
b_nh = pd.read_csv(P / "v2/final/baseline_nohash_5seed.csv").sort_values("doc_id")
h_cl = pd.read_csv(P / "v2/final/htdrop5/oof_ensemble_nested.csv").sort_values("doc_id")
h_nh = pd.read_csv(P / "v2/final/htdrop_nohash_5seed.csv").sort_values("doc_id")
ens = pd.read_csv(P / "v2/final/ens5_confirm/oof_ensemble_nested.csv").sort_values("doc_id")
core = pd.read_csv(P / "data/core_clean_deduped.csv")
CAL = json.load(open(P / "v2/CALIBRATION_RESULTS.json"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- fig 1
def fig_decomposition():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ax = axes[0]
    vals = [0.868, 0.784, 0.542]
    bars = ax.bar(range(3), vals, color=[BASE, MID, FROZ], edgecolor="black", lw=.5, width=.6)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=7.5)
    ax.annotate("", xy=(1, .784), xytext=(0, .868), arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.1))
    ax.text(.5, .905, "discrimination\n0.085 (>=26%)", fontsize=6.6, ha="center")
    ax.annotate("", xy=(2, .542), xytext=(1, .784), arrowprops=dict(arrowstyle="-|>", color="#B22", lw=1.1))
    ax.text(1.55, .655, "calibration\n0.242 (<=74%)", fontsize=6.6, ha="center", color="#B22")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Baseline promotion F1")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Clean\n(oracle thr.)", "Stripped\n(oracle thr.)", "Stripped\n(frozen thr.)"], fontsize=7.5)
    ax.set_title("(a) What the apparent collapse is made of", fontsize=8.5)

    ax = axes[1]
    base = [.785, .784, .542]; htd = [.825, .790, .702]
    x = np.arange(3); w = .35
    b1 = ax.bar(x - w / 2, base, w, label="Baseline", color=BASE, edgecolor="black", lw=.5)
    b2 = ax.bar(x + w / 2, htd, w, label="+ hashtag dropout", color=HTD, edgecolor="black", lw=.5)
    for b in list(b1) + list(b2):
        ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=6.6)
    for i, t in enumerate(["+0.040\n5/5 seeds", "+0.006\n5/5 seeds", "+0.160\nseed CI incl. 0"]):
        ax.text(i, .885, t, ha="center", fontsize=6.4, fontweight="bold")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score, hashtags stripped")
    ax.set_xticks(x)
    ax.set_xticklabels(["AUPRC\n(threshold-free)", "F1\n(oracle thr.)", "F1\n(frozen thr.)"], fontsize=7.5)
    ax.set_title("(b) Effect size depends on the measurement", fontsize=8.5)
    ax.legend(fontsize=6.8, loc="lower right", frameon=False)
    save(fig, "fig1_decomposition")


# ---------------------------------------------------------------- fig 2
def fig_probability_shift():
    """The mechanism: ablation shifts scores under a frozen threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    thr = b_cl[["outer_fold", "nested_threshold_promoting"]].drop_duplicates()
    med = float(np.median(thr.nested_threshold_promoting))
    bins = np.linspace(0, 1, 41)
    for ax, (d, title) in zip(axes, [(b_cl, "Clean input"), (b_nh, "Hashtags stripped")]):
        pos = d.loc[d.label_promoting == 1, "prob_promoting"]
        neg = d.loc[d.label_promoting == 0, "prob_promoting"]
        ax.hist(neg, bins=bins, color="#BBB", alpha=.85, label="promotion negative")
        ax.hist(pos, bins=bins, color=HTD, alpha=.75, label="promotion positive")
        ax.set_yscale("log")
        ax.set_ylim(0.8, 3000)
        ax.axvline(med, color="black", ls="--", lw=1.2)
        ax.axvline(pos.mean(), color="#B22", ls=":", lw=1.4)
        # annotate above the data, in reserved headroom
        ax.annotate(f"frozen thr. {med:.2f}", xy=(med, 900), xytext=(med + .07, 1700),
                    fontsize=6.2, arrowprops=dict(arrowstyle="->", lw=.7, color="black"))
        ax.annotate(f"mean P on\npositives {pos.mean():.3f}", xy=(pos.mean(), 300),
                    xytext=(pos.mean() - .30, 700), fontsize=6.2, color="#B22",
                    arrowprops=dict(arrowstyle="->", lw=.7, color="#B22"))
        below = float((pos < med).mean())
        ax.text(.5, .022, f"{below:.0%} of positives below threshold",
                transform=ax.transAxes, fontsize=6.4, ha="center", style="italic")
        ax.set_title(title, fontsize=8.5, pad=16)
        ax.set_xlabel("P(promotion)")
    axes[0].set_ylabel("documents (log scale)")
    axes[0].legend(fontsize=6.4, frameon=False, loc="upper left", bbox_to_anchor=(0, .82))
    save(fig, "fig2_probability_shift")


# ---------------------------------------------------------------- fig 3
def fig_pr_curves():
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    specs = [(b_cl, BASE, "-", "Baseline, clean"),
             (b_nh, BASE, "--", "Baseline, stripped"),
             (h_nh, HTD, "--", "+dropout, stripped")]
    for d, c, ls, lab in specs:
        pr, rc, _ = precision_recall_curve(d.label_promoting, d.prob_promoting)
        ap = average_precision_score(d.label_promoting, d.prob_promoting)
        ax.plot(rc, pr, color=c, ls=ls, lw=1.4, label=f"{lab} (AP={ap:.3f})")
    # operating points at frozen thresholds
    for d, c, m in [(b_cl, BASE, "o"), (b_nh, BASE, "s"), (h_nh, HTD, "s")]:
        tp = ((d.label_promoting == 1) & (d.pred_promoting == 1)).sum()
        fp = ((d.label_promoting == 0) & (d.pred_promoting == 1)).sum()
        fn = ((d.label_promoting == 1) & (d.pred_promoting == 0)).sum()
        ax.plot(tp / (tp + fn), tp / (tp + fp), m, color=c, ms=5, mec="black", mew=.6, zorder=5)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Promotion label: ranking vs operating point", fontsize=8.5)
    ax.legend(fontsize=6.2, frameon=False, loc="lower left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.text(.42, .12, "markers = frozen\nthreshold operating point", fontsize=6, style="italic")
    save(fig, "fig3_pr_curves")


# ---------------------------------------------------------------- fig 4
def fig_per_stream():
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    streams = [1, 2, 3]
    npos = [int(b_cl[b_cl.stream_group == g].label_promoting.sum()) for g in streams]
    n = [int((b_cl.stream_group == g).sum()) for g in streams]

    def macro(d, g):
        x = d[d.stream_group == g]
        return (f1_score(x.label_suffering, x.pred_suffering, zero_division=0)
                + f1_score(x.label_promoting, x.pred_promoting, zero_division=0)) / 2

    bvals = [macro(b_cl, g) for g in streams]
    evals = [macro(ens, g) for g in streams]
    x = np.arange(3); w = .35
    ax.bar(x - w / 2, bvals, w, label="Baseline", color=BASE, edgecolor="black", lw=.5)
    ax.bar(x + w / 2, evals, w, label="5-encoder ensemble", color=ENS, edgecolor="black", lw=.5)
    for i in range(3):
        ax.annotate(f"{bvals[i]:.3f}", (i - w / 2, bvals[i]), ha="center", va="bottom", fontsize=6.3)
        ax.annotate(f"{evals[i]:.3f}", (i + w / 2, evals[i]), ha="center", va="bottom", fontsize=6.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Stream {g}\nn={n[i]}, {npos[i]} prom+" for i, g in enumerate(streams)], fontsize=6.8)
    ax.set_ylabel("Macro-F1"); ax.set_ylim(0, 1.05)
    ax.set_title("Aggregate scores hide per-stream failure", fontsize=8.5)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax.text(1.0, .06, "streams 2-3 hold only 9 and 2\npromotion positives: not interpretable",
            fontsize=5.8, ha="center", style="italic")
    save(fig, "fig4_per_stream")


# ---------------------------------------------------------------- fig 5
def fig_screening():
    singles = {}
    for name in ["baseline", "bertweet_large", "bertweet_htdrop", "troberta_2022", "deberta_v3"]:
        d = pd.read_csv(P / f"runs_v2/_verify/{name}_2-seed/oof_ensemble_nested.csv")
        singles[name] = (f1_score(d.label_suffering, d.pred_suffering, zero_division=0)
                         + f1_score(d.label_promoting, d.pred_promoting, zero_division=0)) / 2
    ensv = []
    for f in glob.glob(str(P / "runs_v2/_screen_ens/*/oof_ensemble_nested.csv")):
        d = pd.read_csv(f)
        ensv.append((f1_score(d.label_suffering, d.pred_suffering, zero_division=0)
                     + f1_score(d.label_promoting, d.pred_promoting, zero_division=0)) / 2)
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    labels = ["BERTweet-base\n(baseline)", "BERTweet-large", "+hashtag\ndropout", "Twitter-RoBERTa", "DeBERTa-v3"]
    vals = [singles[k] for k in ["baseline", "bertweet_large", "bertweet_htdrop", "troberta_2022", "deberta_v3"]]
    ax.scatter(range(5), vals, s=42, color=BASE, edgecolor="black", lw=.6, zorder=4, label="single systems")
    ax.scatter([5] * len(ensv), ensv, s=22, color=ENS, alpha=.65, edgecolor="black", lw=.4,
               zorder=3, label="26 ensemble subsets")
    ax.axhline(singles["baseline"], color="black", ls="--", lw=.9)
    ax.text(5.4, singles["baseline"] + .0004, "baseline", fontsize=6, ha="right")
    ax.scatter([5], [max(ensv)], s=60, marker="*", color="#B22", edgecolor="black", lw=.5, zorder=5)
    ax.text(5.05, max(ensv) + .0004, "selected", fontsize=6, color="#B22")
    ax.set_xticks(range(6))
    ax.set_xticklabels(labels + ["ensembles"], fontsize=5.8, rotation=25, ha="right")
    ax.set_ylabel("Macro-F1 (2-seed screening)")
    ax.set_title("Selection spread exceeds the confirmed effect", fontsize=8.5)
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    save(fig, "fig5_screening")


# ---------------------------------------------------------------- fig 6
def fig_seed_variance():
    ps = CAL["per_seed_frozen_threshold_delta"]
    seeds = list(ps.keys())
    frozen = [ps[s]["delta"] for s in seeds]
    auprc = [.054463, .043864, .064450, .020706, .026637]
    oracle = [.013030, .017384, .020711, .007814, .018067]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    x = np.arange(len(seeds))
    ax.plot(x, auprc, "o-", color=BASE, lw=1.3, ms=5, label=r"$\Delta$AUPRC")
    ax.plot(x, oracle, "s-", color=ENS, lw=1.3, ms=5, label=r"$\Delta$F1 (oracle thr.)")
    ax.plot(x, frozen, "^-", color=FROZ, lw=1.3, ms=5, label=r"$\Delta$F1 (frozen thr.)")
    ax.axhline(0, color="black", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels(seeds, fontsize=7)
    ax.set_xlabel("Training seed"); ax.set_ylabel(r"$\Delta$ (dropout $-$ baseline)")
    ax.set_title("Only the frozen-threshold gain is seed-unstable", fontsize=8.5)
    ax.legend(fontsize=6.3, frameon=False)
    save(fig, "fig6_seed_variance")


# ---------------------------------------------------------------- fig 7
def fig_dataset():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    ax = axes[0]
    joint = [(core.label_suffering.eq(a) & core.label_promoting.eq(b)).sum() for a, b in
             [(0, 0), (1, 0), (1, 1), (0, 1)]]
    labels = ["neither\n(0,0)", "suffering\nonly (1,0)", "both\n(1,1)", "promotion\nonly (0,1)"]
    bars = ax.bar(range(4), joint, color=["#BBB", BASE, HTD, "#B22"], edgecolor="black", lw=.5)
    for b, v in zip(bars, joint):
        ax.annotate(str(v), (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_ylabel("documents"); ax.set_title("(a) Joint label distribution (n=1,932)", fontsize=8.5)
    ax.text(2.4, 700, "97.6% of promotion positives\nco-occur with suffering", fontsize=6, ha="center", style="italic")

    ax = axes[1]
    streams = [1, 2, 3]
    tot = [int((core.stream_group == g).sum()) for g in streams]
    pos = [int(core[core.stream_group == g].label_promoting.sum()) for g in streams]
    x = np.arange(3); w = .38
    ax.bar(x - w / 2, tot, w, label="all documents", color="#BBB", edgecolor="black", lw=.5)
    ax.bar(x + w / 2, pos, w, label="promotion positive", color=HTD, edgecolor="black", lw=.5)
    for i in range(3):
        ax.annotate(str(tot[i]), (i - w / 2, tot[i]), ha="center", va="bottom", fontsize=6.5)
        ax.annotate(str(pos[i]), (i + w / 2, pos[i]), ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels([f"Stream {g}" for g in streams], fontsize=7.5)
    ax.set_ylabel("documents"); ax.set_title("(b) Promotion concentrates in one stream", fontsize=8.5)
    ax.legend(fontsize=6.3, frameon=False)
    save(fig, "fig7_dataset")


# ---------------------------------------------------------------- fig 8
def fig_reliability():
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    for d, c, ls, lab in [(b_cl, BASE, "-", "Baseline, clean"),
                          (b_nh, BASE, "--", "Baseline, stripped"),
                          (h_nh, HTD, "--", "+dropout, stripped")]:
        p = d.prob_promoting.to_numpy(); y = d.label_promoting.to_numpy()
        order = np.argsort(p); chunks = np.array_split(order, 10)
        xs = [p[i].mean() for i in chunks if len(i)]
        ys = [y[i].mean() for i in chunks if len(i)]
        ax.plot(xs, ys, "o-", color=c, ls=ls, lw=1.2, ms=3.5, label=lab)
    ax.plot([0, 1], [0, 1], color="black", lw=.8, ls=":", label="perfect calibration")
    ax.set_xlabel("mean predicted P(promotion)"); ax.set_ylabel("observed frequency")
    ax.set_title("Reliability: ablation breaks calibration", fontsize=8.5)
    ax.legend(fontsize=6.1, frameon=False, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "fig8_reliability")


if __name__ == "__main__":
    fig_decomposition()
    fig_probability_shift()
    fig_pr_curves()
    fig_per_stream()
    fig_screening()
    fig_seed_variance()
    fig_dataset()
    fig_reliability()
    print("\nAll figures written to", OUT)
