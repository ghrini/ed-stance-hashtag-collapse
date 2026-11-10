# WARNING: `<system>/aggregate/` holds a 3-seed subset, not the paper's numbers

Each `runs_v2/<system>/aggregate/` directory was last written by the
confirmation orchestrator run, which passed `--seeds 71 123 2026`. Those
directories therefore contain **3-seed** aggregates with their own nested
thresholds. They are NOT the artifacts behind any number in the paper, and
their per-fold thresholds differ from the canonical ones.

Using them by mistake changes results materially. For example, applying the
3-seed thresholds to the 5-seed hashtag-stripped predictions yields a
frozen-threshold delta of +0.081 instead of the correct +0.160.

## Canonical artifacts

| Paper content | Canonical file |
|---|---|
| Table II (2-seed screening) | `_verify/<system>_2-seed/oof_ensemble_nested.csv` |
| 5-seed single systems | `_verify/<system>_5-seed/oof_ensemble_nested.csv` |
| 26 screened ensembles | `_screen_ens/<combo>/oof_ensemble_nested.csv` (seeds 13+42) |
| Confirmatory ensemble | `../v2/final/ens5_confirm/oof_ensemble_nested.csv` |
| Hashtag dropout, 5 seeds | `../v2/final/htdrop5/oof_ensemble_nested.csv` |
| Robustness (stripped) | `../v2/final/{baseline,htdrop}_nohash_5seed.csv` |

All of these were built by `../v2/build_ensemble.py` with seeds passed
explicitly. Always control seeds explicitly rather than trusting a directory
name; `build_ensemble.py` takes them as `runs_dir:system:seed1+seed2+...`.
