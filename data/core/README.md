# Core data dictionary

All files here are lightweight derived data or portable parameters. Raw EDF
files, ZIP archives, and neural-network weights are excluded.

## `physical_f/`

- `epoch_features_S_RP_grid.csv.gz`: frozen Zhang+Kumral feature grid used to
  fit the physical densely-repeated proxy. Pandas reads this compressed CSV
  directly.
- `adjacent_repeat_v3_portable.json`: portable parameters for the locked
  physical mapping.
- `all_epoch_trajectories.csv`: physical \(f\) trajectory for every available
  record; the final column used downstream is `f_repeat_micro_v3`.
- `fit_results.json` and the small CSV/Markdown files: fit protocol, nested
  participant-grouped validation, and descriptive diagnostics.

## `staging_inputs/` and `gssc/`

- `kumral_endpoint_labels_66.csv`: one manual 30-s endpoint label for each of
  66 Kumral recordings, with the locked 44/22 train/test split.
- `gssc_endpoint_benchmark_66.csv`: compact endpoint table used to verify exact
  reproduction of the earlier GSSC run.
- `kumral_dominant_band_lookup.csv.gz`: descriptive band labels joined into the
  plotting adapter; never used to tune GSSC or \(g\).
- `gssc/pipeline_stage_trajectories.csv.gz`: final per-epoch five-stage adapter
  used by the REM gate. The production rule is official GSSC argmax, because
  the training-only tuned alternative failed the predeclared stability guard.
- Other `gssc/` files retain endpoint metrics, probability-selection audit,
  and record-level QC.

The Kumral dataset supplies manual ground truth only for the last 30 seconds
of each recording; predicted stages over the earlier epochs are exploratory,
not a manual hypnogram.

## `dream_duration/`, `global_g/`, and `correlations/`

- The consensus table contains the six-LLM duration estimates and the fixed
  analysis split. IDs 1--5 are human calibration seeds and are excluded. The
  primary model additionally excludes short recordings 10, 19, 29, and 33,
  leaving 18 training and 15 held-out records.
- `global_g/` contains all inputs-to-plots produced by the participant-grouped
  nested-CV fit. `all_epoch_trajectories.csv` stores physical \(f\), final
  \(\bar f=g(f)\), GSSC stage, apparent-time increment, and REM-gated dream
  increment for the 37 analyzed reports.
- `correlations/` contains numerical values and QA checks behind the two
  correlation plots. PNG evidence is in `evidence/metrics/`.

The primary stage-weighted formulas are

\[
T_{\rm apparent}=\sum_i g(f_i)\,\Delta t_i,
\qquad
T_{\rm dream}=\sum_i p(\widehat{s_i})\,g(f_i)\,\Delta t_i,
\]

with fixed literature-mean weights
`W=0, N1=0.849, N2=0.532, N3=0.508, REM=0.834`. The weights are not optimized
against the dream-duration targets. The earlier hard REM-only result is kept
only under `historical_baseline/` and is not the primary analysis.
