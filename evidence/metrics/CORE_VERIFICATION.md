# Core reproducibility verification

Verified on 2026-09-05 with the environment documented in
`SOFTWARE_ENVIRONMENT.md`.

## Static and invariant checks

- All five primary command-line modules completed `--help` successfully.
- Compact GSSC inputs contain exactly 66 unique Kumral endpoints with the
  frozen 44 training / 22 test split.
- The GSSC adapter contains 66 records and valid five-stage labels.
- The primary stage-weighted model contains exactly the declared 18 training
  and 15 held-out annotation IDs (33 records, 4,625 epochs).
- Physical `f_repeat_micro_v3` and final `bar_f_lambda` are bounded in [0, 1].
- The portable \(g\) is endpoint-fixed, bounded, and monotone.
- Per-epoch fixed stage weights exactly equal W=0, N1=0.849, N2=0.532,
  N3=0.508, REM=0.834.
- Recomputed record sums of `bar_f_lambda * duration_s` and
  `dream_weight * bar_f_lambda * duration_s` match the archived apparent and
  dream times to better than 1e-7 s.

## Executed numerical smoke and reproduction runs

1. The portable physical predictor read the packaged gzip feature grid and
   reproduced 92 epochs for
   `Kumral:Sub-001_task_sleep-awak003.edf`; its physical \(f\) range was
   0.0038454169865597--0.999944533023969.
2. A complete stage-weighted \(g\) refit was run from the three packaged input
   tables. It again selected `bernstein_degree_5__lambda_0` with the same four
   parameters and identical archived metrics.
3. Numeric values in the refitted training OOF, held-out test, all-record,
   2,001-point \(g\)-curve, and 4,625-row epoch tables had maximum absolute
   difference 0.0 from the archived primary result.
4. The plot program generated 18 training and 15 test sample PNG/CSV pairs,
   both correlation figures, and an index. Its reported metrics were exactly
   equal to the archived correlation metrics; maximum plot-side endpoint
   reconstruction error was 4.55e-13 s.

Full GSSC inference was not repeated during package assembly because it
requires the excluded 46.6-GB raw archive and external pretrained weights.
The packaged GSSC outputs retain the exact 66/66 endpoint hard-label
reproduction check and model/weight provenance needed for that rerun.
