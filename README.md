# TpMT sleep EEG time proxy

Reproducible, exploratory mappings from sleep EEG to phenomenological-time
and stage-weighted dream-duration proxies, inspired by TpMT. This is a
research illustration, **not a validation of TpMT, a measurement of actual
experienced time, or a clinical tool**.

The physical proxy combines multiscale self-similarity with a small
short-lag adjacent-repetition correction. Its calibration uses Zhang and
Kumral EEG. A global endpoint-fixed monotone shell is then calibrated using
Kumral training dream-report duration estimates; GSSC supplies predicted sleep
stages. The primary analysis uses literature-derived stage weights, not a
REM-only gate. Historical REM-only results are retained separately.

## Quick start: redraw without downloading raw EEG or neural weights

Download this repository using GitHub's **Code > Download ZIP**, extract it,
and open a terminal in the extracted directory. Python 3.12 is recommended.
On macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements_figures.txt
python code/plot_kumral_test_sample_figures.py --redraw-only --ids 35 \
  --figure-dir reproduced/figures
```

The output is a PNG/PDF pair for `Sub-006_task_sleep-awak003.edf` (analysis
ID 035; Kumral DREAM Set 19, Case ID 18). Omit `--ids 35` to redraw all 15
retained test examples. On Windows, activate with `.venv\Scripts\Activate.ps1`
in PowerShell; the commands in the full guide use macOS/Linux shell syntax.

## What is included?

| Directory | Contents |
|---|---|
| `code/` | Physical-proxy inference/calibration, global-shell fitting, GSSC adapter, plotting and verification programs |
| `data/core/` | Frozen feature grid, portable model coefficients, split assignments, stage predictions, six-LLM estimates and result tables |
| `data/test_samples/` | Per-epoch trajectories and compact one-second EEG summaries for redraw |
| `figures/test_samples/` | Archived test-example figures |
| `evidence/` | Correlation figures and archived numerical checks |

Large raw EDF/ZIP files, GSSC neural weights, and proprietary LLMs are **not
included**. The EEG display uses within-second summaries of the unfiltered
signal, not a lossless copy of raw EEG. The six-LLM durations are frozen,
subjective screenplay-length estimates, not experimentally observed dream
durations; no LLM API key is needed to replay this analysis.

## Reproduce or extend

- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): installation and commands for
  verification, physical-proxy prediction/refitting, stage inference, shell
  refitting, correlations and figures.
- [DATA_AND_MODELS.md](DATA_AND_MODELS.md): official data/model downloads,
  versions, checksums, storage requirements and provenance.
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md): licenses and citations.

The published lightweight workflow is reproducible **from its frozen feature
grid**. It also provides single-EDF physical-proxy inference and raw Kumral
GSSC inference; it does not claim to automate reconstruction of the complete
historical feature-grid curation or to reproduce fresh LLM judgments exactly.

## Interpretation and evaluation scope

The final dream calibration retains 18 training and 15 held-out records after
the declared exclusions. Held-out Pearson correlations are approximately
0.218 for stage-weighted dream time and 0.196 for total phenomenological time
against report-based duration estimates. These limited associations should
not be described as accurate dream-duration prediction.

Only the final 30-second epoch in each of 66 Kumral records has a manual stage
label in the inputs used here. The held-out endpoint audit has 22 labels:
five-class accuracy is 13/22 (59.1%); REM/non-REM accuracy is 21/22 (95.5%).
This does not validate the full predicted hypnogram, and the held-out audit
contains no wake endpoint and only three REM endpoints.

## License and attribution

Project-authored code is offered under the [MIT License](LICENSE).
Third-party data and their derivatives retain CC BY 4.0 attribution;
GSSC remains AGPL-3.0 and is obtained separately. MIT does not replace those
licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before reuse or
redistribution. Cite the original data and model authors as well as this
repository and the associated TpMT manuscript when using this work.
