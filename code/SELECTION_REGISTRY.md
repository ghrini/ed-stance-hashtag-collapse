# v2 selection registry

Recorded 2026-08-17 ~19:05 local, BEFORE the bertweet_large screening results
were unblinded (its runs were still executing).

## Screening state at time of writing (2-seed ensembles, macro-F1)

- independent baseline (bertweet-base, seeds 13+42): 0.8769
- bertweet_htdrop: 0.8731
- troberta_2022: 0.8721
- deberta_v3: 0.8693
- ens(base+htdrop): 0.8771
- ens(base+htdrop+troberta): 0.8786
- ens(base+htdrop+troberta+deberta): 0.8758
- bertweet_large: NOT YET SEEN
- ensembles including bertweet_large: NOT YET SEEN

## Predeclared rules

1. **Primary candidate selection:** among all screened single systems and
   equal-weight ensembles (members drawn from: baseline, bertweet_htdrop,
   troberta_2022, deberta_v3, bertweet_large), select the configuration with
   the highest 2-seed screening macro-F1.
2. **Confirmation:** every member system of the selected configuration is
   extended to the full five seeds {13,42,71,123,2026}; the confirmatory
   comparison is the 5-seed selected configuration vs the frozen 5-seed
   baseline (macro-F1 0.8740), on identical paired OOF documents, with the
   existing compare_oof.py stratified bootstrap (10k) + paired randomization
   test (100k). One primary metric: macro-F1.
3. **Secondary (robustness) claim, fixed in advance:** bertweet_htdrop vs
   baseline on hashtag-stripped test inputs, promotion F1 — this was already
   observed at screening (0.724 vs 0.562) and will be re-estimated at 5 seeds
   with a paired bootstrap CI.
4. **Guardrails:** suffering-F1 noninferiority margin -0.02 (bootstrap CI
   lower bound); calibration (Brier) must not materially worsen.
5. All screening deltas are exploratory; the paper reports the selection
   process explicitly, including the number of configurations compared, and
   labels remaining optimism honestly.
6. The sealed 800-item challenge is NOT used anywhere in this cycle.

## Amendment 1 (2026-08-17 ~19:4x, recorded BEFORE evaluating dual-view)

Additional exploratory family: dual-view inference = equal-weight average of
each run's clean-text and hashtag-stripped probabilities, using the frozen
clean-validation nested thresholds (no new threshold tuning). Will be
screened at 2 seeds under the same highest-macro-F1 rule alongside the
already-selected all-five ensemble; whichever is higher at screening becomes
the confirmed primary, and this amendment is disclosed in the paper.
