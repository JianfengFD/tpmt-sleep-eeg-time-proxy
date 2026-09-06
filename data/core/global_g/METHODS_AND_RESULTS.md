# Stage-weighted global g calibration

## Definition

The locked physical quantity `f_repeat_micro_v3` is not refitted. The exploratory
GSSC hard stage prediction is mapped to a fixed stage-level probability, and
dream time is

\[
T_{dream}=\sum_i p(\widehat{stage}_i)\,g(f_i)\,\Delta t_i.
\]

This run used the `literature_mean` profile: `W=0, N1=0.849, N2=0.532, N3=0.508, REM=0.834`. These weights are
fixed inputs, not parameters estimated from Kumral. Total apparent time remains

\[
T_{apparent}=\sum_i g(f_i)\,\Delta t_i.
\]

## Records and leakage controls

* IDs 1--5: excluded human-calibration seeds.
* Training IDs 10 and 19: excluded because the recordings are too short.
* Test IDs 29 and 33: excluded because the recordings are too short.
* Remaining training IDs (18 records): model selection and fitting only.
* Remaining held-aside test IDs (15 records): evaluated only after locking g.
* Every inner and outer fold is grouped by participant.
* GSSC stages and the physical f trajectory are frozen upstream inputs and are not
  re-cross-fitted within this conditional validation of g.

Selected configuration: `bernstein_degree_5__lambda_0`.
Decoded parameters: `{"control_points": [0.0, 0.03605627004647669, 0.03605627004647307, 0.03605627004647278, 0.036056270046469524, 1.0], "degree": 5, "interior_control_points": [0.03605627004647669, 0.03605627004647307, 0.03605627004647278, 0.036056270046469524]}`.

## Results

Nested training OOF: n=18, RMSE=683.39 s,
MAE=457.81 s, Pearson r=0.6196,
Spearman rho=0.4628.

Held-aside test: n=15, RMSE=563.27 s,
MAE=434.00 s, Pearson r=0.2179,
Spearman rho=0.1522.

Because `0 <= g <= 1`, `sum p(stage) * duration` is a hard record-level upper
bound. 18/18
training and 14/15 test
targets are at or below that bound.

## Interpretation and limitations

The weights are stage-conditioned priors for obtaining a reportable experience at
an awakening; they are **not** measured, epoch-by-epoch dream occupancy fractions.
The full GSSC trajectory has no manual epoch-wise hypnogram in this dataset. Only
awakening endpoints were checked: five-class endpoint accuracy was 13/22 (59.1%;
balanced accuracy 66.7%), while the REM-versus-non-REM endpoint decision was 21/22
(95.45%). The similar literature priors assigned to N1, N2, N3, and REM reduce the
impact of confusion among sleep subclasses, but the W=0 assumption remains a
potentially sensitive boundary. Results must therefore be described as exploratory
and conditional on the frozen GSSC stage predictions and fixed prior weights.
