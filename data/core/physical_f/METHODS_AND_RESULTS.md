# Fixed R* micro-correction: methods and results

## Definition and scope

The primary mapping is the robust gamma-marginalized self-similarity F_S from v2. Its training N3/W anchors remain mapped to 0.05/0.95. The separate 0.02 in the persistence-ratio formula is only denominator stabilization, not an endpoint target. The new modifier uses P*=q75(P at H=0.5 s and tau=0.08/0.10 s), with NumPy's linear two-value quantile. Inside every fit, the Zhang all-non-W median and W median map smoothly to R=0.05 and 0.95. The physical sign is fixed and every fit must have median(P*|W)>median(P*|non-W).

The final formula is `f=expit(logit(F_S)+0.25*(2R-1))`. Thus recurrence changes the F_S log-odds by at most 0.25 and fixes the exact endpoints. P*, q=.75, and beta=.25 are fixed development choices, not parameters claimed to have been selected by an unbiased nested procedure. Dream reports, dream durations, and spectral/stage-classifier features were not read.

## Validation protocol

Input has 9,197 epochs, 374 records, and 47 subject groups. The outer StratifiedGroupKFold predicts each training subject exactly once. Within each outer fit, the complete robust F_S feature selection and N3/W calibration are rerun, the old P comparator is rerun, and the R* W/non-W calibration uses only outer-training Zhang rows. Test rows never enter fitting.

Because this R* definition was developed after inspection of this study and the held-aside subjects were seen in earlier iterations, the outer result is a cross-fitted development audit and the test result is a repeated held-aside diagnostic—not a pristine confirmatory estimate.

## Cross-fitted training audit

| comparison | pure F_S | old P | fixed R* | R* - pure |
|---|---:|---:|---:|---:|
| W vs N1 | 0.5219 | 0.5607 | 0.5293 | +0.0074 |
| W vs N2 | 0.7358 | 0.7669 | 0.7460 | +0.0102 |
| W vs N3 | 0.8689 | 0.8962 | 0.8743 | +0.0055 |
| W vs REM | 0.6827 | 0.6606 | 0.6837 | +0.0010 |
| W vs all non-W sleep | 0.6562 | 0.6834 | 0.6642 | +0.0080 |

## Repeated held-aside diagnostic

| comparison | pure F_S | old P | fixed R* | R* - pure |
|---|---:|---:|---:|---:|
| W vs N1 | 0.6868 | 0.6730 | 0.6889 | +0.0021 |
| W vs N2 | 0.7278 | 0.7286 | 0.7333 | +0.0056 |
| W vs N3 | 0.9429 | 0.9000 | 0.9571 | +0.0143 |
| W vs REM | 0.7238 | 0.7048 | 0.7333 | +0.0095 |
| W vs all non-W sleep | 0.7181 | 0.7100 | 0.7232 | +0.0050 |

## Endpoint and stability audit

Training nested-OOF medians for the new f are W=0.9598, N1=0.9187, N3=0.0307, and REM=0.7845. The new all-non-W AUC exceeded pure F_S in 5/5 outer folds.

Outer folds selected 2 F_S definitions and 2 old-P definitions. That instability belongs to the freely selected base comparator; P*, q=.75, and beta=.25 do not vary by fold. Exact choices and fold-local anchors are in `selection_stability.csv`.

The old single-P correction gives a larger cross-fitted development gain but loses AUC in the repeated test. R* gives smaller changes but was directionally consistent across the aggregate and the principal N1/REM comparisons in the combined audit. For downstream g fitting, `f_repeat_micro_v3` is the recommended main trajectory; retain pure `f_self_similarity` as the required sensitivity baseline.

## R* weight sensitivity (descriptive only)

`R_weight_sensitivity.csv` compares q=.70/.75/.80/.90 and beta=.20/.25/.275/.5/1.0. For training rows, every q is recalibrated from only the corresponding outer-training Zhang subjects; the repeated-test rows use the full-training calibration. The locked q=.75, beta=.25 setting does avoid AUC decreases for all four W-versus-stage pairs in both summaries. It is a simple, small correction with balanced training/diagnostic behavior; q=.70 and q=.80 are nearby comparators. Larger beta can look stronger in development while reducing held-aside W-N1 or W-REM separation. This post-lock audit was not used to reselect q, beta, F_S, or any model parameter.

## Locked full-training parameters

- F_S: `gamma_median_persistence__long_d9p5__short_d0p75`
- F_S calibration targets: 0.05/0.95 for N3/W anchors
- old P comparator: `recurrence_P_h0p5_tau0p08`, alpha=1
- P* source columns: `recurrence_P_h0p5_tau0p08, recurrence_P_h0p5_tau0p1`
- full-training R* anchors: non-W=-0.09650389, W=0.01899211
- fixed beta: 0.25

## Output map

- `training_nested_oof_predictions.csv`: one leakage-controlled prediction per training endpoint.
- `diagnostic_test_predictions.csv`: locked repeated-test endpoint predictions.
- `all_epoch_trajectories.csv`: locked trajectories for all 9,197 epochs.
- `pairwise_auc_metrics.csv` and `outer_fold_auc_metrics.csv`: aggregate and fold audits.
- `stage_distribution_metrics.csv`: pooled and dataset-specific summaries.
- `subject_cluster_bootstrap_summary.csv`: subject-level uncertainty audit.
- `R_weight_sensitivity.csv`: post-lock q/beta robustness audit; never a selector.
- `adjacent_repeat_v3_portable.json`: parameters needed for later application and g fitting.
- `PREDICTING.md` and `code/predict_adjacent_repeat_v3.py`: standalone CSV/EDF application.
