# Third-party notices and citations

Checked against the official distribution records on 2026-09-06.
The root MIT license applies to project-authored software and documentation,
not to third-party data, report text, software or model weights. No original
data collection, neural-model authorship or endorsement is claimed.

## Data: CC BY 4.0

The following sources use [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
Retain attribution, source and license links, and identify modifications when
redistributing the source material or adaptations. The rights of the original
contributors are unchanged by this repository.

1. **Zhang & Wamsley 2019 Final**, deposited by Erin Wamsley, Jing Zhang and
   Megan Collins (2023), figshare, version 1.
   [Data](https://doi.org/10.6084/m9.figshare.22226692.v1).
   Related paper: Zhang J, Wamsley EJ (2019), *EEG predictors of dreaming
   outside of REM sleep*, Psychophysiology 56, e13368.
   [Article](https://doi.org/10.1111/psyp.13368).
2. **Daytime experiences shape neural activity and dream content**, Deniz
   Kumral, Jessica Palmieri, Steffen Gais and Monika Schönauer (2024),
   FreiData v1, DREAM Set 19; the archive's historical name is
   `Kumral et al., 2023.zip`.
   [Data](https://freidata.uni-freiburg.de/records/31mg4-mfq53).
   [Machine-readable license metadata](https://freidata.uni-freiburg.de/api/records/31mg4-mfq53).
   Related paper: Kumral D et al. (2025), *Pre-sleep experiences shape neural
   activity and dream content in the sleeping brain*, iScience 28, 113032.
   [Article](https://doi.org/10.1016/j.isci.2025.113032).
3. **The DREAM database**, William Wong, Thomas Andrillon, Nicolas Decat,
   Rubén Herzog, Valdas Noreika, Katja Valli, Jennifer Windt and Naotsugu
   Tsuchiya; metadata version 9.
   [Data](https://doi.org/10.26180/22133105.v9).
   Related paper: Wong W et al. (2025), *A dream EEG and mentation database*,
   Nature Communications 16, 7495.
   [Article](https://doi.org/10.1038/s41467-025-61945-1).

**Modifications here:** recordings/metadata were selected, aligned and
processed into self-similarity/repetition features, stage predictions,
integrated proxy trajectories and one-second EEG summaries. Public English
dream report text is reproduced in the compact tables and in figures users
can generate locally (pre-rendered figures are not published here);
derived LLM duration columns are added by this project and are not annotations
supplied by the original investigators. Subject/record IDs are retained for
traceability. The data-derived portions of `data/`, `evidence/` and locally generated figures
must retain this CC BY 4.0 attribution. Do not attempt participant
re-identification. Report contents may describe personal experiences.

## GSSC: AGPL-3.0, downloaded separately

**Greifswald Sleep Stage Classifier (GSSC), version 0.0.9**, copyright
University Clinic Greifswald, Germany, distributed under
[GNU Affero General Public License v3](https://github.com/jshanna100/gssc/blob/main/LICENSE).
[Official source](https://github.com/jshanna100/gssc) ·
[Pinned PyPI release](https://pypi.org/project/gssc/0.0.9/).

Citation: Hanna J, Flöel A (2023), *An accessible and versatile deep
learning-based sleep stage classifier*, Frontiers in Neuroinformatics 17,
1086634. [Article](https://doi.org/10.3389/fninf.2023.1086634).

No GSSC source or neural weights are bundled. The project's adapter calls the
upstream implementation and follows its official loudest-vote rule. Its MIT
notice does not relicense GSSC. If you distribute a combined or modified GSSC
application, comply with the applicable AGPL requirements; separation of
downloads is not a general exemption from those requirements. Preserve the
upstream license and notices when downloading or redistributing GSSC.

Other Python dependencies are installed separately from their official
packages. NumPy, pandas, SciPy, scikit-learn, MNE-Python, PyTorch, Matplotlib,
joblib and importlib-resources each retain their own upstream licenses and
notices; the dependency lists are not license replacements.

## Stage-weight source and LLM estimates

Stucky B (2025), *We Are the Sensors of Consciousness! A Review and Analysis
on How Awakenings During Sleep Influence Dream Recall*, Nature and Science
of Sleep 17, 709–729. [Article](https://doi.org/10.2147/NSS.S506461).
Detailed definitions and supporting references are in
`data/core/global_g/LITERATURE_STAGE_WEIGHTS.md`. These weights are awakening
report rates used as operational priors, not observed fractions of time
spent dreaming.

The consensus CSV identifies the six estimation sources and preserves their
values. No proprietary LLM code, weights or access credentials are distributed.
The archived numerical judgments can be replayed without querying an LLM;
new calls may produce different judgments.
