# Kumral GSSC stage-trajectory and endpoint audit

## Five-class endpoint result used to characterize the stage model

The paper-candidate analysis uses the official GSSC five-class hard vote to
assign W, N1, N2, N3, or REM to each modeled 30-s epoch. Only the final epoch
of each Kumral record has a public manual stage label. At those endpoints,
five-class accuracy was 26/44 (59.1%) in the participant-grouped training
partition and 13/22 (59.1%) in the held-out partition. Held-out class-wise
correct counts were N1 1/2, N2 7/14, N3 2/3, and REM 3/3; there were no
manually labeled W endpoints. These endpoint figures do not validate the
earlier full-record stage trajectory.

## Secondary REM/non-REM endpoint audit

The binary audit rule is the official GSSC five-class argmax collapsed to
**REM versus non-REM**. A train-only search over 20 genuine-probability score/smoothing
candidates selected `consensus__trailing_mean_2` at threshold
`0.0554688852`, but it did not pass the subject-bootstrap replacement guard.
Across 919 usable out-of-bag bootstrap replicates, 20-candidate in-bag
reselection had median balanced-accuracy gain 0.000 over official argmax, 95%
percentile interval [-0.071, 0.400], and was better in only 28.9% of replicates.
The tuned threshold is also low and unstable (2.5%--97.5% bootstrap range
0.055--0.500).  It is exported for audit, but is not used downstream.

At the 22 held-out endpoints (six unseen subjects), the locked binary REM gate
gave TN=18, FP=1, FN=0, TP=3: accuracy 95.45% (Wilson 95% CI
78.20%--99.19%), REM sensitivity 100% (43.85%--100%), non-REM specificity
94.74% (75.36%--99.06%), balanced accuracy 97.37%, REM F1 85.71%, and ROC
AUC 96.49%.  The intervals are necessarily wide because the test set contains
only three manually labelled REM endpoints.

Pooling all 66 labelled endpoints gives the descriptive confusion counts
TN=55, FP=2, FN=2, TP=7 (accuracy 93.94%, balanced accuracy 87.13%).  This
pooled number includes the training/model-selection partition and is therefore
**not** the primary held-out estimate; it is provided only as a descriptive
summary.

For comparison, the experimental tuned gate reduced held-out accuracy to
86.36% and REM precision to 50%; this is another reason it was rejected.

## Probability semantics and validation scope

GSSC produces five-class log-softmax outputs for C3-TP10 and C4-TP9 after
0.3--30 Hz filtering, 85.33 Hz resampling, continuous per-channel z-scoring,
and full-record bidirectional-GRU context.  The official `loudest_vote` rule
selects, at each epoch, the channel permutation having the largest maximum
log-probability. The upstream run also produced a larger diagnostic table of
per-channel probability variants; it is not needed by the final analysis and
is intentionally not included here. `pipeline_stage_trajectories.csv.gz`
retains the normalized official five-class probabilities and selected hard
stage required downstream. No hard-label one-hot vectors are presented as
probabilities.

The Kumral release provides only one manual stage label: the final 30-second
epoch of each EDF.  Therefore the reported accuracy validates **endpoint
REM/non-REM discrimination only**.  The other 8,515 epochs have no manual
hypnogram in this release.  Their stages remain model predictions, not measured
ground truth.  `record_level_qc.csv` describes REM fractions, bout structure,
confidence/entropy, and C3/C4 disagreement so implausible whole-record ribbons
can be flagged, but these diagnostics cannot substitute for manual scoring.

The final epoch hard label reproduced the earlier GSSC benchmark for all 66/66
records exactly. All 8,581 probability epochs align one-to-one with the locked
physical trajectory keys, including each record's non-zero prefix. The 37
pre-exclusion dream-report candidates contribute 4,724 verified epoch keys;
after the four short-record exclusions, the primary 18/15 analysis uses 33
records and 4,625 epochs.

## Files

- `pipeline_stage_trajectories.csv.gz`: downstream adapter.  It has the exact
  join fields `record_uid, epoch_index, start_s, end_s, duration_s`, locked
  `predicted_stage`, genuine official `p_W...p_REM`, and the carried-forward
  `dominant_band`.
- `endpoint_predictions.csv`: all 66 endpoint labels and locked/experimental
  predictions.
- `metrics.json`: locked, baseline, experimental, confidence-interval, and
  alignment results.
- `train_only_candidate_selection.csv`: subject-LOSO ranking of all 20
  candidates.
- `train_subject_bootstrap_stability.csv.gz`: 1,000 train-subject bootstrap
  replicates (996 valid in-bag; 919 usable OOB comparisons).
- `record_level_qc.csv`: unlabelled whole-record diagnostic summary.
- `manifest.json`: model, preprocessing, leakage exclusions, and provenance.

## Reproduce

Run from the candidate-package root after placing the external Kumral archive
and GSSC source/weights as described in `REPRODUCIBILITY.md`:

```bash
export PYTHONPATH="$PWD/code/core"
python -m analysis.sleep_staging.kumral_gssc_rem_probability \
  --archive '/absolute/path/Kumral et al., 2023.zip' \
  --labels data/core/staging_inputs/kumral_endpoint_labels_66.csv \
  --benchmark data/core/staging_inputs/gssc_endpoint_benchmark_66.csv \
  --physical-trajectories data/core/physical_f/all_epoch_trajectories.csv \
  --old-stage-trajectories data/core/staging_inputs/kumral_dominant_band_lookup.csv.gz \
  --dream-consensus data/core/dream_duration/kumral_dream_duration_6llm_consensus.csv \
  --output /tmp/tpmt_gssc_recomputed \
  --temp-dir /tmp
```

Neither dream-report text nor estimated dream duration, filename, recording
duration, or subject identity is used as a classifier feature or threshold
target.  Held-out test labels are evaluated only after the production rule is
locked from training subjects.
