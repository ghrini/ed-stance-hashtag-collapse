# Quantifying and Repairing Hashtag-Ablation Collapse in Eating-Disorder Stance Detection

Code, per-run predictions, and result files for the IEEE IEMCON 2026 paper of the same name.

**Author:** Ghrini Sai Jella (Independent Researcher) — ghrinisai.jella@gmail.com

## What the paper shows

A BERTweet-base classifier trained to separate *suffering* (disclosure of one's own struggle)
from *promotion* (content encouraging disordered eating) on 1,932 deduplicated tweets scores
promotion-F1 0.862. When hashtags are removed from test input, promotion-F1 falls to 0.542.

- At most 74% of that collapse is decision-threshold miscalibration; at least 26% is real loss of discriminative signal.
- Re-selecting the threshold on hashtag-ablated *validation* data, with no retraining, restores promotion-F1 to 0.759 (+0.217, 95% CI [0.174, 0.261]; 90% of the oracle gap). A label-free variant restores 0.752.
- The repair is replicated with a strictly held-out validation set (+0.199, 91% of its own oracle gap).
- The repair is not free: false promotion flags rise from 39 to 126, mostly on posts that also disclose suffering.
- None of six model-side alternatives (dependency heads, TAPT, weak supervision, encoder substitution, five-encoder ensemble, hashtag-dropout augmentation) gave a significant gain; the one apparent exception disappears when stratified by hashtag presence.

All evaluation is development cross-validation (5 folds x 5 seeds). The confirmatory challenge set is not used anywhere in this repository.

## Layout

| Path | Contents |
|---|---|
| `code/` | Training, evaluation, repair and analysis scripts (`recalibration_recovery.py`, `genuine_heldout_recalibration.py`, `labelfree_thresholds.py`, `stratify_hashtag_analysis.py`, ...) |
| `results/json/` | Canonical result files behind every number in the paper |
| `results/oof/` | Out-of-fold predictions (doc id, fold, labels, probabilities, thresholds). No tweet text. |
| `results/ALL_NUMBERS.md` | Every reported number with its source file |
| `figures/` | `fig_robustness.*` is the paper figure; the rest are exploratory |

Start with `results/READ_THIS_BEFORE_USING_AGGREGATES.md`: some intermediate aggregate folders are stale subsets and must not be used.

## Data

The tweets are **not** redistributed here. The corpus is the public eating-disorder tweets dataset of
Benítez-Andrades et al. (see the paper's reference list). Obtain it from its original source and follow
its license and X's terms. The out-of-fold files key rows by `doc_id`/`text_hash`/`cluster_id`, so the
fold assignment can be reconstructed from `results/oof/baseline_independent_oof.csv`.

## Reproducing

- `pip install -r requirements.txt`
- The scripts were flattened from the original project layout and expect `data/core_clean_deduped.csv` and `artifacts/dev_outer_folds_v1.csv` relative to the repository root; a few contain absolute paths from the author's machine (`make_figures.py`, `results/json/FINAL_RESULTS.json`) that you will need to change.
- Statistics (paired stratified bootstrap and sign-flip randomization) can be re-run from the OOF CSVs without a GPU.

## License

MIT for the code (see `LICENSE`). The underlying corpus has its own terms.

## Citation

If you use this work, please cite the IEMCON 2026 paper (full citation to be added once the proceedings are indexed).
