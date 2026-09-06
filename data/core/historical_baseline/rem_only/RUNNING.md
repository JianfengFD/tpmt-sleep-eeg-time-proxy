# Reproduce REM-only g calibration

```bash
cd /Users/lijf/Documents/work/CODEX_PRJS/TpMT_Sleep
XDG_CACHE_HOME=.cache_runtime MPLCONFIGDIR=.cache_runtime/matplotlib \
  .venv/bin/python -m analysis.combined_fawake.fit_rem_only_global_g
```

The command writes only to `outputs/dense_repeat_fawake_v3/rem_only_g_gssc_v2_confirmed` and does not alter the locked physical-f
or exploratory-stage source files.
