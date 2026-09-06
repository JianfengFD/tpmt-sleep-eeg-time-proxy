# Applying the locked preferred R* model

`predict_adjacent_repeat_v3.py` is the standalone application entry point for
the locked physical mapping

\[
P^*=q_{0.75}(P_{H=.5,\tau=.08},P_{H=.5,\tau=.10}),\qquad
f=\operatorname{expit}[\operatorname{logit}(F_S)+0.25(2R^*-1)].
\]

It consumes the fitted anchors and the complete robust self-similarity model
from `adjacent_repeat_v3_portable.json`. The predictor rejects a JSON whose
fixed q, beta, lag columns, endpoint calibration, or physical direction has
been changed.

## Apply to one record already present in an epoch-feature CSV

From the project root:

```bash
.venv/bin/python -m analysis.combined_fawake.predict_adjacent_repeat_v3 \
  --model-json outputs/dense_repeat_fawake_v3/rstar_micro_fit/adjacent_repeat_v3_portable.json \
  --epoch-features outputs/dense_repeat_fawake_v2/epoch_features_S_RP_grid.csv \
  --record-uid 'Kumral:Sub-001_task_sleep-awak003.edf' \
  --output /tmp/kumral_awak001_preferred_f.csv
```

If the CSV contains exactly one record, omit `--record-uid`. Use the physical
feature-grid CSV here, not the compact `all_epoch_trajectories.csv` result
(which intentionally omits the 18 selected raw S columns). The input must
contain `duration_s`, the self-similarity columns named by the portable model,
and the two locked P columns. Extra columns—including dream text, duration
estimates, stages, or an earlier prediction—are ignored by the numerical path.

## Apply directly to a raw EDF

```bash
.venv/bin/python -m analysis.combined_fawake.predict_adjacent_repeat_v3 \
  --model-json outputs/dense_repeat_fawake_v3/rstar_micro_fit/adjacent_repeat_v3_portable.json \
  --edf /absolute/path/to/record.edf \
  --dataset Kumral \
  --metadata-duration-s 930 \
  --record-uid 'new-record' \
  --output /tmp/new_record_preferred_f.csv
```

EDF mode uses the shared channel selection, filtering/resampling, robust epoch
normalization, and complete 30-second epochs aligned backward from the valid
signal end. If `--metadata-duration-s` is omitted, the EDF header duration is
used. Supplying the dataset metadata duration is recommended when it differs
from the header because it determines the valid endpoint and alignment.

## Outputs and scope

The epoch CSV exposes raw/calibrated `F_S`, the old-P comparator, raw/calibrated
`R*`, `f_repeat_micro_v3`, and the per-epoch pure/preferred apparent-time
increments. A sibling `.summary.json` reports both integrals and all source
metadata needed for an audit.

This program stops before the dream-calibrated global `g`. It does not read a
dream report or dream-duration target, does not infer sleep stages, and does
not require a stage label. Apply the separately fitted global g only in the
next pipeline layer.
