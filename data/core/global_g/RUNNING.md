# Reproduce stage-weighted g calibration

Run from the paper-candidate package root:

```bash
export PYTHONPATH="$PWD/code/core"
python -m analysis.combined_fawake.fit_stage_weighted_global_g \
  --consensus-csv data/core/dream_duration/kumral_dream_duration_6llm_consensus.csv \
  --physical-epochs data/core/physical_f/all_epoch_trajectories.csv \
  --stage-epochs data/core/gssc/pipeline_stage_trajectories.csv.gz \
  --f-column f_repeat_micro_v3 \
  --output /tmp/tpmt_global_g_refit \
  --outer-splits 5 --inner-splits 4 \
  --regularization-grid '0,0.001,0.01,0.1,1,10' \
  --weight-profile literature_mean
```

To rerun with the originally requested heuristic weights, use
`--weight-profile requested_heuristic`. To use revised literature values, supply
all five values with `--stage-weights` and an explicit `--output`. The command
above writes only to `/tmp/tpmt_global_g_refit`; it does not modify the archived
primary result or historical REM-only baseline. See the package-level
`REPRODUCIBILITY.md` for the complete pipeline and verification commands.
