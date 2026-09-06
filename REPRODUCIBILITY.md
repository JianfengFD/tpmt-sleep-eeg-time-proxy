# Reproducing the TpMT EEG sleep-proxy analysis

This package preserves the complete lightweight analysis chain from a
physical EEG proxy \(f\), through external GSSC stage inference, to one global
endpoint-fixed monotone \(g\), fixed stage-weighted dream time, correlations,
and plots.
It deliberately excludes large raw recordings and pretrained neural weights.
Every command below writes to a new directory and leaves the archived results
unchanged.

## 0. Install the environment

Start in the downloaded/extracted repository root. The archived run used
macOS arm64 and Python 3.12.4. Use Python 3.12 for the pinned environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/core/requirements.txt
export MPLCONFIGDIR=/tmp/tpmt_mpl_cache
export PYTHONDONTWRITEBYTECODE=1
```

For figure-only redraw, the smaller `code/requirements_figures.txt` is enough;
that environment does not run refits or raw-EEG inference. Full direct package
versions are pinned; transitive dependencies and platform/font differences
can still affect byte-for-byte output. If the PyTorch 2.5.0 wheel is unavailable
on your platform, consult the official PyTorch installation instructions and
record any substitute version; do not describe it as the exact archived
environment. This project does not require an LLM API key.

## 1. Package layout and integrity

From this package directory:

```bash
export PKG="$PWD"
export PYTHONPATH="$PKG/code/core"
python code/core/verify_core_package.py --package-root "$PKG" --verify-manifest
```

`FILE_MANIFEST.csv` contains byte sizes and SHA-256 hashes for every regular
file except the manifest itself. Regenerate it only after intentionally
changing the package:

```bash
python code/core/build_manifest.py --package-root "$PKG"
```

## 2. Required external inputs

To rerun raw-signal steps, obtain these files separately:

Official download commands, sizes, hashes, licenses, and the GSSC installation
command are in [DATA_AND_MODELS.md](DATA_AND_MODELS.md). Original authors and
paper citations are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

1. Kumral et al. (2023) archive, expected filename
   `Kumral et al., 2023.zip`, 46,588,397,863 bytes. Public record:
   <https://freidata.uni-freiburg.de/records/31mg4-mfq53>.
2. Zhang and Wamsley (2019) archive, expected filename
   `Zhang_Wamsley_2019_Final.zip`, 1,049,159,479 bytes. Public record:
   <https://figshare.com/articles/dataset/Zhang_Wamsley_2019_Final/22226692>.
3. GSSC 0.0.9 source plus its official `sig_net_v1.pt` and `gru_net_v1.pt`
   weights. Place or symlink the installed tree at
   `code/core/.vendor/gssc`. Exact weight hashes are in
   `code/core/.vendor/README.md`.

The archived package starts from the frozen feature grid, so neither raw
archive is needed to refit \(f\), fit \(g\), recompute correlations, or make
plots. The Kumral archive and GSSC weights are needed only to regenerate stage
probabilities. An individual EDF is needed only when applying \(f\) directly
to a new record.

## 3. Physical densely-repeated EEG proxy

The locked physical mapping uses a robust self-similarity component and a
small adjacent-return-prominence correction. It does not read sleep stages,
dream text, or dream-duration targets.

Apply the portable mapping to one record already in the frozen feature grid:

```bash
python -m analysis.combined_fawake.predict_adjacent_repeat_v3 \
  --model-json "$PKG/data/core/physical_f/adjacent_repeat_v3_portable.json" \
  --epoch-features "$PKG/data/core/physical_f/epoch_features_S_RP_grid.csv.gz" \
  --record-uid 'Kumral:Sub-001_task_sleep-awak003.edf' \
  --output /tmp/tpmt_one_record_f.csv
```

Apply the same mapping directly to an EDF:

```bash
python -m analysis.combined_fawake.predict_adjacent_repeat_v3 \
  --model-json "$PKG/data/core/physical_f/adjacent_repeat_v3_portable.json" \
  --edf /absolute/path/to/record.edf \
  --dataset Kumral \
  --metadata-duration-s 930 \
  --record-uid new-record \
  --output /tmp/tpmt_new_record_f.csv
```

To refit the physical proxy from the frozen Zhang+Kumral grid using the
declared 5x4 participant-grouped nested validation and 2,000 subject-cluster
bootstrap replicates:

```bash
python -m analysis.combined_fawake.fit_adjacent_repeat_v3 \
  --features "$PKG/data/core/physical_f/epoch_features_S_RP_grid.csv.gz" \
  --output /tmp/tpmt_physical_refit \
  --outer-splits 5 --inner-splits 4 --bootstrap-replicates 2000
```

## 4. GSSC five-stage inference and endpoint audit

GSSC consumes complete end-aligned Kumral sequences using C3-TP10 and
C4-TP9, a 0.3--30 Hz bandpass, 85.333 Hz resampling, and continuous
per-channel z-scoring. The official per-epoch loudest-vote argmax is the
production rule. Only the final 30-s epoch of each of the 66 recordings has a
manual stage label; the 44 training and 22 held-out endpoint labels are never
expanded into a full manual hypnogram.

With the external archive and GSSC tree in place:

```bash
python -m analysis.sleep_staging.kumral_gssc_rem_probability \
  --archive '/absolute/path/Kumral et al., 2023.zip' \
  --labels "$PKG/data/core/staging_inputs/kumral_endpoint_labels_66.csv" \
  --benchmark "$PKG/data/core/staging_inputs/gssc_endpoint_benchmark_66.csv" \
  --physical-trajectories "$PKG/data/core/physical_f/all_epoch_trajectories.csv" \
  --old-stage-trajectories "$PKG/data/core/staging_inputs/kumral_dominant_band_lookup.csv.gz" \
  --dream-consensus "$PKG/data/core/dream_duration/kumral_dream_duration_6llm_consensus.csv" \
  --output /tmp/tpmt_gssc_recomputed \
  --temp-dir /tmp
```

This processes 66 EDFs and can take substantial time. Its output adapter is
`pipeline_stage_trajectories.csv.gz`. The archived adapter and endpoint audit
are already present under `data/core/gssc/`.

## 5. Fit the primary stage-weighted global monotone g

IDs 1--5 are human calibration seeds and are excluded. Short recordings 10,
19, 29, and 33 are also excluded by the fixed protocol. The resulting 18
training IDs and 15 held-out IDs are listed in the portable JSON. Physical
\(f\) and GSSC stages are frozen before this step. The stage is not an input
to \(g\); it selects a fixed literature-derived dream probability:

```bash
python -m analysis.combined_fawake.fit_stage_weighted_global_g \
  --consensus-csv "$PKG/data/core/dream_duration/kumral_dream_duration_6llm_consensus.csv" \
  --physical-epochs "$PKG/data/core/physical_f/all_epoch_trajectories.csv" \
  --stage-epochs "$PKG/data/core/gssc/pipeline_stage_trajectories.csv.gz" \
  --f-column f_repeat_micro_v3 \
  --output /tmp/tpmt_global_g_refit \
  --outer-splits 5 --inner-splits 4 \
  --regularization-grid '0,0.001,0.01,0.1,1,10' \
  --weight-profile literature_mean
```

Candidate endpoint-fixed monotone families are selected by minimum mean
participant-grouped cross-validated log-MSE on the training set only. The
fixed weights are `W=0, N1=0.849, N2=0.532, N3=0.508, REM=0.834`; they are
not optimized against the reports. In the archived run, the selected model
was `bernstein_degree_5__lambda_0`, an endpoint-fixed, bounded, monotone
four-parameter shell.

## 6. Recompute correlation and per-record plots

The plotting program reads only archived global-g tables and the consensus
table. It writes both correlations plus all 18 training and 15 test record
plots and replayable per-record CSVs:

```bash
python -m analysis.combined_fawake.plot_stage_weighted_g_results \
  --root "$PKG/data/core/global_g" \
  --consensus "$PKG/data/core/dream_duration/kumral_dream_duration_6llm_consensus.csv" \
  --output-dir /tmp/tpmt_final_plots \
  --dpi 190
```

The archived numerical correlation tables are under
`data/core/correlations/`. Figure files are not published in this repository;
the command above generates the corresponding PNGs locally.
The held-out test Pearson correlations are 0.218 for stage-weighted predicted
dream time versus estimated report duration and 0.196 for total apparent time
versus estimated report duration (15 retained tests). These are exploratory
calibration results, not a validation of the TpMT theory.

## 7. Redraw the paper-candidate test figures without raw EDFs

The manuscript-facing figure program is separate from the full scientific
plotter above. It reads only the packaged trajectory and one-second EEG
summary CSVs under `data/test_samples/`; neither the 46.6-GB archive nor the
upstream project is accessed in redraw-only mode. From the candidate root:

```bash
python code/plot_kumral_test_sample_figures.py --redraw-only \
  --figure-dir /tmp/tpmt_paper_figures
```

This writes the 15 retained test-record PNG/PDF pairs to a separate output
directory. No pre-rendered figure files are included in the public repository.
To redraw only ID 35:

```bash
python code/plot_kumral_test_sample_figures.py --redraw-only --ids 35 \
  --figure-dir /tmp/tpmt_paper_id35
```

The corresponding compact numerical inputs remain available for inspection
and independent plotting. The full scientific plotting command in section 6
continues to generate correlations and all 18 training plus 15 test records.

## 8. Rebuild compact staging inputs from the original project

This optional provenance step requires the original upstream result tree:

```bash
python code/core/prepare_core_data.py \
  --source-root /absolute/path/to/TpMT_Sleep \
  --output /tmp/tpmt_staging_inputs
```

It filters exactly 66 Kumral endpoint labels, the matching 66 earlier GSSC
benchmark rows, and the 8,581-row dominant-band lookup without touching any
scientific outputs.

## 9. Common errors and limitations

- `No module named analysis`: activate the environment and export
  `PYTHONPATH="$PWD/code/core"` from the repository root.
- Missing `.vendor/gssc`: follow DATA_AND_MODELS.md and check the weight
  hashes. The model is intentionally not distributed with this repository.
- Missing original `outputs/...` or `data/...`: legacy module defaults refer
  to the original project. Use the explicit paths in the commands above.
  The manuscript plotter must use `--redraw-only` unless original inputs are
  supplied via `--analysis-root`, `--consensus`, and `--kumral-zip`.
- Do not infer a record's true stage from its filename, or propagate its
  final manual label across all earlier epochs. Full-record stage labels are
  predictions, not manual ground truth.
- Refits must keep the declared subject groups and training/test assignments.
  Report-based durations are noisy proxy targets, not measured dream time.
- The lightweight release starts from the frozen feature grid; it does not
  provide a turnkey re-creation of every historical raw-data curation step.
