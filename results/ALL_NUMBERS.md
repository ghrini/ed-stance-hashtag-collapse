# Every verified number, with provenance

All results are **development-set** 5-fold cross-validation with seed
ensembling. Nothing here touched the sealed 800-item challenge set.
Every comparison is **paired on the same 1,932 documents**.

---

## 1. Dataset

Source: Benítez-Andrades et al. ED tweets corpus, deduplicated to one
representative per near-duplicate cluster.

| Quantity | Value |
|---|---|
| Documents | 1,932 |
| Suffering positive / negative | 954 / 978 |
| Promoting positive / negative | 456 / 1,476 |
| Joint (suffering, promoting) = (0,0) | 967 |
| (1,0) suffering only | 509 |
| (1,1) both | 445 |
| (0,1) promotion without suffering | 11 |
| Promotion positives co-occurring with suffering | 445/456 = 97.6% |
| Stream 1 / 2 / 3 documents | 1,273 / 638 / 21 |
| Promotion positives by stream | 445 / 9 / 2 |

Source file: `results/oof/core_clean_deduped.csv`.

---

## 2. Protocol (needed to describe Methods accurately)

- Encoder: `vinai/bertweet-base` (baseline), two independent sigmoid heads on
  the [CLS] representation, mean BCE loss.
- Hyperparameters: max_length 128, dropout 0.15, batch 16 × 2 gradient
  accumulation, encoder LR 2e-5 (1e-5 for BERTweet-large), head LR 1e-4,
  weight decay 0.01, warmup ratio 0.1, **4 epochs**, AMP mixed precision.
- Folds: one frozen 5-fold outer manifest from a deterministic greedy
  allocator that jointly balances both labels, the rare joint states, and
  stream membership.
- Thresholds: tuned by F1 grid search (0.05–0.95, step 0.01) on an **inner**
  80/20 validation split only; never on outer-fold test items. Final
  decisions use per-fold **nested** thresholds re-tuned on seed-averaged
  inner-validation predictions.
- Training budget: identical **four exposures per example** for every system
  (compute-matched).
- Seeds: {13, 42} for screening; {13, 42, 71, 123, 2026} for confirmation.
- Inference statistics: stratified bootstrap (10,000 replicates, resampling
  within fold × stream strata) + paired sign-flip randomization test
  (100,000 permutations). Seeds/folds are never treated as independent
  observations.
- Hardware/software: NVIDIA RTX PRO 4500 Blackwell 32 GB, torch 2.11.0+cu128,
  transformers 5.15.0, Python 3.11.9.

---

## 3. Baseline (BERTweet-base, independent heads, 5 seeds)

Source: `results/oof/baseline_independent_oof.csv`,
`results/json/independent_vs_dependency_hard.json` (baseline block).

| Metric | Value |
|---|---|
| **Macro-F1** | **0.8740205** |
| Suffering F1 | 0.8860501 |
| Promoting F1 | 0.8619910 |
| Suffering AUPRC | 0.9464992 |
| Promoting AUPRC | 0.9394079 |
| Macro AUPRC | 0.9429535 |
| Suffering Brier | 0.0796471 |
| Promoting Brier | 0.0478456 |
| Suffering ECE | 0.0352533 |
| Promoting ECE | 0.0213512 |
| Exact match | 0.8281573 |
| Hamming loss | 0.0892857 |

### Per-stream baseline (this is the shortcut evidence)

| Stream | n | Macro-F1 | Suffering F1 | Promoting F1 |
|---|---|---|---|---|
| 1 | 1,273 | 0.8924 | 0.9108 | 0.8740 |
| 2 | 638 | 0.5926 | 0.8322 | **0.3529** |
| 3 | 21 | 0.1667 | 0.3333 | **0.0000** |

---

## 4. NEGATIVE RESULT 1 — dependency-aware architectures

The promotion logit receives a learned scalar × the *detached* predicted
suffering probability. Variants add TF-IDF-mined hard or random contrastive
triplets on suffering-positive anchors.

Source: `results/json/independent_vs_dependency*.json`.

| System | Macro-F1 | Δ vs baseline | 95% CI | p |
|---|---|---|---|---|
| Dependency | 0.8733387 | −0.0006818 | [−0.00439, 0.00307] | .724 |
| Dependency + random contrastive | 0.8717363 | −0.0022842 | [−0.00801, 0.00347] | .443 |
| Dependency + hard contrastive | 0.8754190 | +0.0013984 | [−0.0046134, 0.0074895] | .651 |

Note for the paper: dependency-hard was selected as the best of three, so its
point estimate carries winner's-curse optimism. Also, an independent code
audit found the hierarchy penalty disabled in training, hard mining
restricted to suffering-positive anchors only, and the contrastive term
applied on alternating microbatches (roughly half its nominal weight) —
these are documented in `background/INDEPENDENT_AUDITS.md`.

---

## 5. NEGATIVE RESULT 2 — task-adaptive pretraining (TAPT)

5,000 MLM steps on a 457,775-line leakage-filtered in-domain tweet corpus
(effective batch 64, LR 5e-5, all layers trainable), then identical
supervised training.

Source: `results/json/base_independent_vs_tapt_independent.json`.

| Metric | Value |
|---|---|
| TAPT macro-F1 | 0.8054357 |
| Δ vs baseline | **−0.0685848** |
| 95% CI | [−0.0843122, −0.0538722] |
| p | ≈ 1.0e-5 |
| Δ promoting F1 | −0.1011456 |
| Δ macro AUPRC | ≈ −0.051 |

Both labels' Brier scores worsened. TAPT-dependency and TAPT-independent both
converged near 0.805, indicating the adapted encoder — not the head — caused
the failure. Provenance caveat to state honestly: the TAPT run's corpus hash
and trainer version have documented inconsistencies
(`background/INDEPENDENT_AUDITS.md`), so the correct claim is "this specific
TAPT pipeline/checkpoint is harmful," not "TAPT cannot work here."

---

## 6. NEGATIVE RESULT 3 — weak supervision (prior project iteration)

Accuracy-weighted labeling functions over an unlabeled pro-ED Reddit pool
(~52K deduplicated posts), confident-example selection.

Source: `results/json/legacy_cv_v2_summary.csv`,
`results/json/legacy_weak_supervision_v2_results.csv`.

| Evaluation | Suffering Δ | Promoting Δ | p (promoting) |
|---|---|---|---|
| Single split (misleading) | +0.00218 | **+0.029099** | n.s., CI [−0.032, 0.091] |
| Matched-compute 3-fold × 2-seed CV | −0.009704 (p=.165) | **+0.003191** | **.504** |

Manual weak-label audit: ~77% accuracy overall, ~70% among weak
promotion-positive cases. The single-split "improvement" did not replicate —
this is the paper's case study in why single-split evaluation misleads.

---

## 7. Screening pass (2 seeds, 13/42)

Source: `results/json/screening_2seed.json`. Baseline row uses the same two
seeds for a like-for-like comparison.

| System | Macro-F1 | Suffering F1 | Promoting F1 |
|---|---|---|---|
| BERTweet-base (baseline, 2-seed) | 0.8768945 | 0.8921819 | 0.8616071 |
| BERTweet-large | 0.8736551 | 0.8964814 | 0.8508287 |
| BERTweet-base + hashtag dropout | 0.8730731 | 0.8843398 | 0.8618063 |
| Twitter-RoBERTa-base-2022 | 0.8721209 | 0.8941300 | 0.8501119 |
| DeBERTa-v3-base | 0.8692507 | 0.8914168 | 0.8470847 |
| **Ensemble of all five** | **0.8809** | 0.8955 | 0.8662 |

**No single alternative encoder beat the domain-matched baseline at the
screening seeds.** All 26 ensemble subsets were screened; the all-five
ensemble won under a selection rule predeclared before BERTweet-large's
results were seen (`code/SELECTION_REGISTRY.md`).

### IMPORTANT — single systems re-scored at 5 seeds (transparency)

The single-system ordering is **not stable** across seed counts, which the
paper states explicitly:

| System | 2-seed macro-F1 | 5-seed macro-F1 |
|---|---|---|
| Baseline (BERTweet-base) | 0.8768945 | 0.8740205 |
| BERTweet-large | 0.8736551 | **0.8749425** |
| Twitter-RoBERTa-2022 | 0.8721209 | 0.8740520 |
| BERTweet-base + hashtag dropout | 0.8730731 | 0.8700780 |
| DeBERTa-v3-base | 0.8692507 | 0.8679123 |

At five seeds BERTweet-large edges the baseline by +0.0009 and
Twitter-RoBERTa ties it — differences far inside seed noise. **Do not write
"no encoder beats the baseline" without the seed qualifier.** The honest
claim is that individual encoder differences are within noise, while the
ensemble gain and the robustness effect are not.

### Rejected variant: dual-view inference

Averaging each model's clean and hashtag-stripped probabilities at test time.
Screened and rejected (registry Amendment 1, recorded before evaluation):

| Dual-view system | Macro-F1 |
|---|---|
| Baseline | 0.8396 |
| + hashtag dropout | 0.8615 |
| Twitter-RoBERTa | 0.7923 |
| DeBERTa-v3 | 0.8467 |
| BERTweet-large | 0.8196 |
| Five-system ensemble | 0.8525 |

All worse than the selected ensemble (0.8809).

---

## 8. CONFIRMATORY RESULT — five-encoder ensemble (5 seeds)

Source: `results/json/ens5_vs_baseline.json`,
`results/oof/ens5_confirm_oof.csv`.

| Metric | Baseline | Ensemble |
|---|---|---|
| **Macro-F1** | 0.8740205 | **0.8778199** |
| Suffering F1 | 0.8860501 | 0.8931535 |
| Promoting F1 | 0.8619910 | 0.8624862 |
| Macro AUPRC | 0.9430 | **0.9483** |
| Suffering Brier | 0.0796 | **0.0741** |
| Promoting Brier | 0.0478 | **0.0471** |

| Comparison | Δ | 95% CI | p |
|---|---|---|---|
| **Macro-F1** | **+0.0037994** | **[−0.0034288, 0.0111888]** | **.311** |
| Suffering F1 | +0.0071 | [−0.0009, 0.0153] | — |
| Promoting F1 | +0.0005 | [−0.0116, 0.0129] | — |

Suffering noninferiority guardrail (margin −0.02): **passed**.

### Per-stream deltas — the gains land where the baseline fails

| Stream | n | Baseline macro-F1 | Ensemble macro-F1 | Δ |
|---|---|---|---|---|
| 1 | 1,273 | 0.8924 | 0.8978 | +0.0054 |
| 2 | 638 | 0.5926 | 0.6146 | **+0.0220** |
| 3 | 21 | 0.1667 | 0.5000 | **+0.3333** |

**Honest framing required in the paper:** the screening estimate was +0.0040
and the confirmatory estimate is +0.0038, but versus the 5-seed baseline the
aggregate delta is **not statistically significant** (p = .311). The paper
reports it as a consistent improvement across accuracy, calibration, AUPRC,
and worst-stream performance — not as a significant one.

---

## 9. ROBUSTNESS — CORRECTED (read this section carefully)

> **An earlier version of this document and of the paper reported a
> +0.160 promotion-F1 robustness gain as the headline significant result.
> That framing was WRONG and has been corrected.** Independent review found
> that the comparison applied thresholds tuned on clean validation data to
> hashtag-stripped test data, which conflates calibration shift with
> discriminative loss. Verified source: `v2/CALIBRATION_RESULTS.json`,
> produced by `code/calibration_analysis.py`.

### 9-PRE. Two corrections made after independent review (2026-08-17, later pass)

1. **Precision under ablation.** An earlier draft claimed clean precision was
   0.825 and "unchanged" under ablation. **Wrong.** Verified values:
   clean P=0.8902 R=0.8355 (TP 381, FP 47, FN 75, 428 flags);
   stripped P=0.8251 R=0.4035 (TP 184, FP 39, FN 272, 223 flags).
   Precision falls 0.065; the real story is that **predicted-positive volume
   drops 48%** (428 → 223). Never write "precision unchanged".
2. **The "best threshold" is an ORACLE** selected on the evaluation data.
   Therefore 74% is an **upper bound** on the calibration share and 26% a
   **lower bound** on the discriminative share. On clean data the oracle buys
   +0.006 over the nested threshold (0.868 vs 0.862). Always label it.

### 9-SEED. Seed-level analysis (this changes which claims are robust)

Document-level bootstrapping conditions on one fitted ensemble. Refitting per
seed gives a very different picture of which effects are stable:

| Seed | ΔAUPRC | ΔF1 (oracle) | ΔF1 (frozen) |
|---|---|---|---|
| 13 | +0.0545 | +0.0130 | +0.2197 |
| 42 | +0.0439 | +0.0174 | +0.1004 |
| 71 | +0.0644 | +0.0207 | +0.0886 |
| 123 | +0.0207 | +0.0078 | +0.0559 |
| 2026 | +0.0266 | +0.0181 | +0.0087 |
| **mean** | **+0.0420** | **+0.0154** | **+0.0947** |
| **95% t-CI** | **[+0.0192, +0.0649]** | **[+0.0091, +0.0217]** | **[−0.0027, +0.1921]** |

- ~~AUPRC gain is robust — positive in 5/5 seeds, seed-CI excludes zero.~~
  **RETRACTED, see 9-RETRACTED below.** Seed-consistency of a pooled number
  is not the same as a real per-document effect.
- ~~Oracle-F1 gain is also robust per seed (+0.015)~~ **RETRACTED, same
  reason.**
- **The +0.160 frozen-threshold delta is NOT seed-robust** (CI includes zero).
  Do not present it as a robustness claim; it is descriptive only.

### 9-RETRACTED. The AUPRC "positive result" does not survive stratification

> **The paper has NO surviving positive result as of the current draft.**
> This was found in a third review round, verified independently, and is
> documented in full in `../REVIEW_HISTORY.md` Round 3. Source:
> `code/stratify_hashtag_analysis.py` → `results/json/STRATIFICATION_RESULTS.json`.

Of the 1,932 documents, 1,058 contain a hashtag (ablation changes their
text) and 874 do not (ablation is a no-op: stripped text is byte-identical
to clean text). Splitting the AUPRC and oracle-F1 comparisons by this
variable:

| Stratum | n (pos.) | ΔAUPRC [95% CI] | ΔF1 oracle [95% CI] |
|---|---|---|---|
| Pooled (Table IV / what was reported) | 1,932 (456) | +0.0402 [0.025, 0.055] | +0.0059 [−0.005, 0.020] |
| Had hashtag (ablation active) | 1,058 (351) | +0.0002 [−0.009, 0.009] | −0.0043 [−0.010, 0.006] |
| No hashtag (ablation inert) | 874 (105) | −0.0036 [−0.025, 0.017] | +0.0016 [−0.026, 0.028] |

**Both strata are null for both metrics.** The pooled gain is a
between-group pooling artifact — the two strata have different promotion
prevalence (0.33 vs 0.12) and different score scales, so pooling them into
one ranking/threshold can move the metric even though nothing changed
within either group. Per-seed stratified checks (5 seeds × 2 strata) show
no seed with a stable within-stratum effect; what looked like "5/5-seed
consistency" was consistency of the pooling artifact across seeds, not of a
real per-document effect.

**This also answers the paper's circularity question**, previously left
open: the 874 hashtag-free documents (105 positives) ARE the natural,
non-circular test the paper wanted (data where the shortcut is absent
rather than artificially removed), at far higher power than the
9-promotion-positive stream-2 subset originally proposed for this. The
answer is null.

### 9-RECAL. The remedy that survived (the paper's one positive result)

Source: `code/recalibration_recovery.py` -> `results/json/RECALIBRATION_RESULTS.json`
Checks: `REVIEW_HISTORY.md` Round 3 addendum.

Procedure: reconstruct each fold's original inner-validation split, run the
saved v1 baseline checkpoints on hashtag-STRIPPED validation text, tune a new
promotion threshold on the seed-ensembled stripped validation predictions
(never touching outer-test labels), apply per-fold to the stripped test set.

| Quantity | Value |
|---|---|
| Frozen-threshold stripped F1 | 0.5420 |
| **Recalibrated stripped F1** | **0.7591** |
| Delta (paired, stratified bootstrap) | **+0.2171, 95% CI [0.1741, 0.2609]** |
| Oracle ceiling | 0.7837 |
| Fraction of oracle gap recovered | **89.8%** |
| Recalibrated thresholds by fold | 0.18 / 0.18 / 0.19 / 0.18 / 0.23 (frozen were 0.47-0.74) |

Robustness checks (both passed, with the signature a real effect should have):

| Stratum | n (pos.) | frozen -> recal | delta [95% CI] |
|---|---|---|---|
| Had hashtag (ablation active) | 1,058 (351) | 0.5433 -> 0.8442 | +0.3009 [0.2555, 0.3532] |
| No hashtag (ablation inert) | 874 (105) | 0.5385 -> 0.5560 | +0.0175 [-0.0607, 0.0979] (null; unharmed) |

| Seed | frozen -> recal | delta |
|---|---|---|
| 13 | 0.4618 -> 0.7379 | +0.2761 |
| 42 | 0.5645 -> 0.7585 | +0.1941 |
| 71 | 0.5299 -> 0.7203 | +0.1904 |
| 123 | 0.6028 -> 0.7664 | +0.1636 |
| 2026 | 0.6124 -> 0.7445 | +0.1321 |

Seed-level t-CI: **[+0.1248, +0.2578]** (positive 5/5). Recalibration also
shrinks the across-seed spread of stripped F1 threefold (0.151 -> 0.046).

Caveats (stated in the paper): exploratory/post hoc (direct test of the
decomposition's implied remedy, not a searched-for hypothesis); presupposes
knowing which perturbation to calibrate against.

### 9-TRAP. Threshold provenance

`runs_v2/<system>/aggregate/` contains a **3-seed** aggregate (seeds
71/123/2026) written by the confirmation orchestrator. Applying its
thresholds to 5-seed predictions gives +0.081 instead of +0.160. The
canonical 5-seed artifact is `v2/final/htdrop5/`. See
`runs_v2/READ_THIS_BEFORE_USING_AGGREGATES.md`.

### 9a. The decomposition (baseline, promotion label)

| Quantity | Promotion F1 |
|---|---|
| Clean input, best threshold | 0.8684 |
| Hashtags stripped, best threshold | 0.7837 |
| Hashtags stripped, **frozen** clean threshold | 0.5420 |
| Loss attributable to **calibration shift** | 0.2417 (**74%**) |
| Loss attributable to **discrimination** | 0.0847 (**26%**) |

Under ablation the baseline's precision is *unchanged* (0.825 → 0.825);
only recall collapses (0.835 → 0.404). A genuine loss of signal would
degrade both. The frozen threshold (per-fold median ≈0.53) ends up above
the shifted probability mass (mean P on true positives = 0.454), producing
272 mechanical false negatives.

### 9b. Is the shortcut real? YES — threshold-free evidence

| Baseline, clean minus stripped | Estimate | 95% CI |
|---|---|---|
| AUPRC | **−0.1547** | [0.1182, 0.1911] |
| Best-threshold F1 | **−0.0847** | [0.0609, 0.1118] |

Both exclude zero. The shortcut exists; it costs about **10% of achievable
promotion F1**, not the 37% the naive measurement implies.

### 9c. Does hashtag dropout help? Modestly, and only on one measure

| Measurement (hashtags stripped) | Base | +Dropout | Δ [95% CI] |
|---|---|---|---|
| AUPRC (threshold-free) | 0.7847 | 0.8249 | **+0.0402 [0.0247, 0.0551]** significant |
| F1 at best threshold | 0.7837 | 0.7896 | +0.0059 [−0.0047, 0.0196] **null** |
| F1 at frozen threshold | 0.5420 | 0.7022 | +0.1602 [0.1267, 0.1954] (calibration stability) |

**Costs (clean input), which must be reported:**

| | Base | +Dropout | Δ [95% CI] |
|---|---|---|---|
| Promotion AUPRC | 0.9394 | 0.9334 | −0.0060 [−0.0107, −0.0020] **significantly worse** |
| Suffering AUPRC | — | — | −0.0090 [−0.0164, −0.0016] **significantly worse** |
| Suffering Brier | 0.0796 | 0.0822 | worse |
| Promotion Brier | 0.0478 | 0.0507 | worse |
| Promotion ECE | 0.0214 | 0.0327 | **+53% worse** |

By the project's own predeclared guardrail ("calibration must not materially
worsen"), the standalone hashtag-dropout system **fails its guardrail**.
Report it as a robustness/calibration trade-off, never as a free win.

**Stability:** the frozen-threshold delta ranges **+0.009 to +0.220** across
single seeds (mean +0.095) versus a bootstrap CI half-width of ±0.034 —
document resampling alone understates uncertainty ~3×.

**Circularity caveat:** the entire gain lives in shortcut-saturated stream 1
(0.551 → 0.715) and does not transfer to naturally hashtag-sparse stream 2
(0.250 → 0.222, only 9 promotion positives). Training with p=0.5 hashtag
deletion moves the training distribution toward the all-deleted test
distribution, so some of the gain is distribution matching. State this.

---

## 9-OLD. Superseded framing (kept only to prevent reuse)

Training-time augmentation: each hashtag independently deleted with
probability 0.5. Test inputs unmodified. Diagnostic: same models re-evaluated
on test text with **all hashtags stripped**, using each system's own frozen
nested thresholds.

Source: `results/json/FINAL_RESULTS.json` (robustness block),
`results/oof/baseline_nohash_5seed.csv`, `results/oof/htdrop_nohash_5seed.csv`.

| System | Clean suff. F1 | Clean prom. F1 | Stripped suff. F1 | Stripped prom. F1 |
|---|---|---|---|---|
| Baseline | 0.8861 | 0.8620 | 0.8746 | **0.5420** |
| + hashtag dropout | 0.8846 | 0.8556 | 0.8780 | **0.7022** |

| Comparison | Δ | 95% CI |
|---|---|---|
| **Promoting F1, hashtags stripped** | **+0.1601811** | **[0.1269052, 0.1956048]** |

Cost on clean inputs (5-seed, paired vs baseline): **−0.0039425 macro-F1**,
95% CI [−0.0116, 0.0035], **p = .301** → statistically null. Suffering
noninferiority: passed.

Interpretation for the paper: **more than a third of the baseline's promotion
detection is hashtag-dependent** (0.862 → 0.542). Hashtag dropout recovers
most of that gap at no measurable clean-test cost. At screening, every
alternative encoder showed the same vulnerability (Twitter-RoBERTa fell to
0.436 promoting F1), so the shortcut is a property of the data, not of one
architecture — it requires a training-time fix, not a better encoder.

---

## 10. Numbers that appear in the paper's tables

- Table I (negative results): section 4, 5, 6 above.
- Table II (screening): section 7 above.
- Table III (confirmatory): section 8 above.
- Table IV / shortcut table: section 9 above.
- Figure 1: promotion-F1 clean vs stripped, both systems, with the
  +0.160 [0.127, 0.196] annotation.

---

## 11. Known limitations to keep in the paper

1. Small (1,932), English-only, Twitter-only, acquisition-biased corpus;
   stream 3 has 21 documents and supports no reliable estimate.
2. **Author identifiers are unavailable** in the released data, so folds are
   document-grouped, not author-grouped; repeated authors may cross folds.
   Confidence intervals are conditional on the fitted folds.
3. The suffering label may have been annotated with access to author profile
   context not visible in the single tweet — a construct-validity risk
   flagged by an independent audit and not yet resolved.
4. Screening compared 5 single systems + 26 ensembles + 6 dual-view variants,
   so the selected configuration retains mild selection optimism even though
   the confirmatory comparison was prespecified.
5. All results are development cross-validation. No sealed/held-out
   evaluation has been run.
6. TAPT provenance inconsistencies (corpus hash, trainer version) limit the
   TAPT claim to this specific pipeline.
