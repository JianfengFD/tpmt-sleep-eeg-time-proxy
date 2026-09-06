# REM-only global g calibration

## Definition

The locked physical quantity `f_repeat_micro_v3` is never refitted.  The exploratory
epoch-wise stage table is joined to it exactly on `record_uid, epoch_index, start_s, end_s, duration_s`.  Dream
time is defined as

\[
T_{dream}=\sum_i 1[\widehat{stage}_i=REM]\,g(f_i)\,\Delta t_i,
\]

whereas total apparent time is reported separately as

\[
T_{apparent}=\sum_i g(f_i)\,\Delta t_i.
\]

The stage labels are exploratory classifier predictions, not manual epoch-wise
hypnograms.  They gate the REM-only dream integral but do not modify physical f.

## Leakage controls and fitting

* IDs 1--5: excluded human calibration seeds.
* IDs 6--25: training and model selection only.
* IDs 26--42: held-aside diagnostic evaluation after locking g.
* Every outer and inner fold is grouped by participant.
* The nested OOF result is conditional validation of `g`: its selection and fitting
  exclude each validation participant, but the already frozen upstream physical-f
  and exploratory stage models are not re-cross-fitted inside this run.  It must not
  be described as fully end-to-end OOF performance.
* Candidate g families have zero through four free parameters and are monotone,
  bounded in [0,1], with exact endpoints g(0)=0 and g(1)=1.
* Selection minimizes participant-balanced inner-CV log-MSE of REM-gated dream
  time.  Raw-second MAE/RMSE and correlations are reported without entering the
  selection rule.

Selected configuration: `identity__lambda_0`.
Decoded parameters: `{}`.

## Results

Nested training OOF dream-time result: n=20, RMSE=988.90 s,
MAE=638.80 s, Pearson r=0.3218,
Spearman rho=0.4513.

Held-aside test dream-time result: n=17, RMSE=742.54 s,
MAE=607.27 s, Pearson r=0.3522,
Spearman rho=0.4453.

REM-duration feasibility is a hard upper bound because 0 <= g <= 1.  Only
13/20 training and
11/17 test targets are at or below
their predicted REM duration.  This limitation is preserved rather than hidden.
