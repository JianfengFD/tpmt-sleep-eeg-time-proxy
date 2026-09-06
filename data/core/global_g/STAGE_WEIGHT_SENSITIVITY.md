# Stage-weight sensitivity comparison

The stage-specific literature profile was fixed as the primary analysis before
examining the held-out test results. The other profiles are sensitivity analyses
only; their test metrics were not used to select the reported primary prior.

All runs exclude training IDs 10 and 19 and test IDs 29 and 33. Training values
are conditional participant-grouped nested OOF estimates for `g`; test values use
one global `g` selected and fitted using the remaining training records only.

| Profile | Role | Weights W/N1/N2/N3/REM | Selected g | Train r / rho | Test r / rho | Test RMSE / MAE (min) |
|---|---|---|---|---:|---:|---:|
| `literature_stage_specific` | primary_prespecified | 0.000/0.849/0.532/0.508/0.834 | `bernstein_degree_5__lambda_0` | 0.620 / 0.463 | 0.218 / 0.152 | 9.39 / 7.23 |
| `literature_nrem_rem` | sensitivity_only | 0.000/0.595/0.595/0.595/0.834 | `bernstein_degree_5__lambda_0` | 0.610 / 0.463 | 0.206 / 0.152 | 9.53 / 7.20 |
| `requested_heuristic` | sensitivity_only | 0.000/0.350/0.150/0.000/0.850 | `bernstein_degree_5__lambda_1` | 0.609 / 0.580 | 0.390 / 0.376 | 8.80 / 7.25 |

## Interpretation

The literature stage-specific and combined-NREM profiles give very similar
held-out results, so N1/N2/N3 confusions are not driving the main result when
their literature weights are close. The originally requested heuristic gives
a higher test correlation in this small sample, but it is not supported as a
population mean by the reviewed awakening literature. Promoting it on the basis
of these test results would constitute post-hoc test-set selection.

The weak held-out correlations across all three profiles show that the result is
exploratory and prior-sensitive. These comparisons do not validate the stage
weights as instantaneous dream-occupancy probabilities.
