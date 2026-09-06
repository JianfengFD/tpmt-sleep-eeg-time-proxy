# Running the fixed R* micro-correction audit

From the project root:

```bash
XDG_CACHE_HOME=.cache_runtime MPLCONFIGDIR=.cache_runtime/matplotlib \
.venv/bin/python -m analysis.combined_fawake.fit_adjacent_repeat_v3 \
  --features outputs/dense_repeat_fawake_v2/epoch_features_S_RP_grid.csv \
  --output outputs/dense_repeat_fawake_v3/rstar_micro_fit \
  --outer-splits 5 \
  --inner-splits 4 \
  --bootstrap-replicates 2000
```

The input must contain the complete self-similarity grid, the return-prominence
grid, and the declared train/test metadata. The script does not read dream
reports or durations and does not fit g. The output directory is independent
of v2 and never overwrites it.

Use `predict_adjacent_repeat_v3.py` with the exported portable JSON to apply
the locked mapping to one existing feature-table record or directly to a raw
Zhang/Kumral EDF. See `PREDICTING.md` in the output directory.
