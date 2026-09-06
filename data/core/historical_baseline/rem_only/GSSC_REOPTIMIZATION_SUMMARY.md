# Confirmed GSSC-only re-optimization of the global g

## Was the previous g already fitted with GSSC?

Yes.  The preceding `rem_only_g_gssc_v1` fit used
`outputs/sleep_staging_kumral_rem_probability_v1/pipeline_stage_trajectories.csv.gz`,
whose REM gate is the official GSSC five-class argmax on C3-TP10/C4-TP9.  The
subsequent Kumral accuracy audit did not change that trajectory; it established
stronger evidence for the already-used gate.

This directory is a clean rerun after the accuracy audit.  It is outside the
`paper_candidate_kumral_rem_gate` folder, which was not modified by this rerun.

## Re-optimized g

The physical `f_repeat_micro_v3` was kept fixed.  Training IDs 6--25 alone
selected among the endpoint-fixed, bounded, monotone zero- through four-parameter
shells.  The participant-grouped cross-validation criterion again selected

```text
identity__lambda_0:  g(x) = x
```

No test target entered the model selection or fitting.

## Updated train/test results

| Quantity compared with six-LLM dream-duration estimate | Split | n | MAE (s) | RMSE (s) | Pearson r | Spearman rho |
|---|---|---:|---:|---:|---:|---:|
| REM-gated predicted dream time | Training conditional nested OOF | 20 | 638.80 | 988.90 | 0.322 | 0.451 |
| REM-gated predicted dream time | Held-out test | 17 | 607.27 | 742.54 | 0.352 | 0.445 |
| Total apparent time | Training conditional nested OOF | 20 | 1811.53 | 2410.13 | 0.292 | 0.357 |
| Total apparent time | Held-out test | 17 | 1505.28 | 1781.86 | 0.070 | 0.117 |

The test Pearson p-values are 0.166 for REM-gated dream time and 0.790 for
total apparent time.  The corresponding Spearman p-values are 0.073 and 0.656.

## Exact comparison with the preceding run

The following files were compared between `rem_only_g_gssc_v1` and this clean
rerun: `g_curve.csv`, `heldout_test_predictions.csv`,
`training_nested_oof_predictions.csv`, `all_final_model_record_predictions.csv`,
and `all_epoch_trajectories.csv`.  Shapes and columns matched, and the maximum
absolute numeric difference was 0.0.  The two plot-metric JSON objects were
also exactly equal.

Thus the accuracy audit requires no numerical correction to `g`: it validates
the GSSC gate already used.  The remaining weakness is the REM-only calibration
itself: four of 17 held-out records contain no predicted REM, and only 11 of 17
held-out dream-duration targets are no larger than the total predicted REM
duration, which is a hard upper bound because `0 <= g <= 1`.

## Updated figures

- `final_plots_v2/predicted_dream_time_vs_estimate.png`
- `final_plots_v2/apparent_time_vs_estimate.png`
- `final_plots_v2/training_samples/`: 20 record-level trajectories
- `final_plots_v2/test_samples/`: 17 record-level trajectories
- `final_plots_v2/combined_correlation_values.csv`: source values for both
  correlation figures

