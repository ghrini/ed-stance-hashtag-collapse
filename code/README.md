# ED-STANCE v2

New-encoder / shortcut-robustness experiments layered on the frozen v1
protocol. Nothing in the parent directory is modified; v2 runs live in
`../runs_v2/`.

## What is new versus v1

| Piece | v1 | v2 |
|---|---|---|
| Encoders | BERTweet-base only | + BERTweet-large, Twitter-RoBERTa-base-2022, DeBERTa-v3-base |
| Heads | independent + dependency variants | independent only (dependency was a null result) |
| Augmentation | contrastive triplets (null) | hashtag dropout p=0.5 (each hashtag independently deleted during training) |
| Diagnostics | none | every run also predicts its test fold with all hashtags stripped (`predictions_nohash.csv`) |
| Ensembling | seeds within one system | `build_ensemble.py` averages any set of (system, seed) members with nested re-tuned thresholds |

Folds, schedule (4 epochs, batch 16 x 2 accumulation, warmup 0.1), inner-split
threshold tuning, and OOF pairing are byte-identical to v1, so every v2
system pairs item-by-item with the frozen baseline.

## Files

- `train_fold_v2.py` — one (system, fold, seed) run. Per-encoder LR is a CLI
  argument recorded in `run.json`.
- `run_screen_v2.py` — resumable orchestrator; system specs (encoder, LR,
  hashtag dropout) are declared in `SPECS`.
- `aggregate_oof_v2.py` — seed-ensemble + nested thresholds + provenance
  checks; also aggregates the no-hashtag diagnostic.
- `build_ensemble.py` — cross-system / seed-subset ensembles (equal-weight
  probability averaging, thresholds re-tuned on identically averaged
  inner-validation predictions only).
- `baseline_nohash_predict.py` — replays the saved v1 baseline checkpoints on
  hashtag-stripped test text so the baseline appears in the shortcut table.
- `screen_summary.py` — one-line-per-system screening table.
- `config_v2.json` — shared hyperparameters (mirrors `../config.json`).

## Protocol guardrails carried over from v1

- Thresholds are only ever tuned on inner-validation predictions (per fold,
  after seed averaging) — never on outer-fold test items.
- Screening used seeds {13, 42}; the selected system is confirmed with all
  five seeds {13, 42, 71, 123, 2026} before any comparison is quoted.
- Paired comparisons (stratified bootstrap + sign-flip randomization) run on
  identical OOF documents via `../compare_oof.py`.
- The sealed 800-item challenge is untouched by everything in this directory.
