# Historical baseline (not the primary analysis)

This directory preserves the earlier hard-REM-only calibration for audit
continuity. Its model was the identity shell and its report counts were 20
training plus 17 test records. It is superseded by the primary
stage-weighted analysis in `../global_g/`, which uses fixed literature-derived
weights, removes four predeclared too-short recordings, and selects a
degree-five monotone Bernstein shell on training records only.

Do not use `rem_only/portable_rem_only_global_g.json` as the paper's primary
model. The REM-only source modules remain in `code/core` because the current
stage-weighted implementation reuses their general input-joining and metric
utilities.
