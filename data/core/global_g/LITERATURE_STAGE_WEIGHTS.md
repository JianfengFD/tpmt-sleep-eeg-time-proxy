# Sleep-stage dream-report rates and operational weights for the stage-weighted `g` model

**Research question.** Can the proposed stage probabilities `N1 = 0.35`, `N2 = 0.15`, and `REM = 0.85` be treated as literature-based probabilities of dreaming, and what fixed stage weights should be used in

\[
\widehat T_{\mathrm{dream}}
=\int_0^T p\{S(t)\}\,\bar f(t)\,dt,
\qquad \bar f(t)=g\{f(t)\}?
\]

## Bottom line

- **REM = 0.85 is well supported** as an immediate-awakening *dream/experience-with-content report rate*. Recent evidence gives 0.81--0.90, and the 69-study mean is 0.834.
- **N1 = 0.35 is not supported** as a broad dream-experience or content-recall rate. The 69-study mean is 0.849; the DREAM database raw pooled rate is 0.88; individual serial-awakening studies give roughly 0.75--0.77 (and sometimes higher).
- **N2 = 0.15 is not supported** as a broad dream-experience or content-recall rate. The 69-study mean is 0.532; DREAM gives 0.56; relevant individual studies give roughly 0.42--0.68.
- The best defensible single, externally fixed operational prior is therefore

  \[
  (p_W,p_{N1},p_{N2},p_{N3},p_{REM})
  =(0,0.849,0.532,0.508,0.834),
  \]

  or, rounded for prose, `(0, 0.85, 0.53, 0.51, 0.83)`.
- `p_W = 0` is a **target-definition choice** (waking mentation is not counted as dream time), not the empirical wake report rate. Likewise, forcing `p_N3 = 0` would be a structural restriction, not a conclusion supported by the broad dream-report literature.
- These values should be called **literature-derived stage reportability/recallability weights**, not direct measurements of the fraction of every sleep stage spent dreaming. They are conditional probabilities from awakening protocols.

## What exactly was measured?

The literature does not observe dreaming continuously. Its main measurement is a verbal report immediately after an experimentally scheduled or spontaneous awakening. Three outcomes must be kept distinct:

1. **Experience with recall / content recall (E or CE):** the participant reports some content. In modern broad protocols this can include a thought, isolated image, perception, emotion, or immersive narrative dream.
2. **Experience without recall (EWR/CEWR; “white dream”):** the participant is convinced that an experience occurred but cannot retrieve its content.
3. **No experience / no report (NE/NCE):** no experience is reported. This is not proof that no experience occurred; encoding, retrieval, sleep inertia, question wording, and response criteria can all create a null report.

“Dream recall” is also protocol-dependent. A liberal prompt such as “What was going through your mind just before the alarm?” detects more minimal mentation than the stricter “What did you dream?” A delayed morning diary is not equivalent to an immediate serial awakening. Therefore, rates from different definitions cannot be silently mixed.

## 69-study review: study-level means

Stucky (2025) reviewed **69 healthy-participant awakening studies from 2000--2024** and separated experience with recall, experience without recall, and no report. The following are the published **means of available study percentages**, rather than a single participant-level pooled binomial estimate.

| Sleep stage | Experience with recall | Experience without recall | No report |
|---|---:|---:|---:|
| NREM, pooled | 59.5% | 23.8% | 28.0% |
| N1 | **84.9%** | 10.0% | 2.0% |
| N2 | **53.2%** | 22.0% | 35.9% |
| N3 | **50.8%** | 28.9% | 21.1% |
| REM | **83.4%** | 7.1% | 13.2% |
| Wake | 95.3% | 6.6% | 3.6% |

**Important denominator warning:** each outcome and stage was available in a different subset of studies. The three columns are separately averaged study percentages, so rows are not expected to add to exactly 100%. In particular, do **not** estimate “any experience” by simply adding the first two mean columns.

The review also reports that recall depends on awakening method, study duration, laboratory versus home setting, and participant differences; NREM estimates are especially sensitive. Thus 0.849/0.532/0.508/0.834 are useful fixed population-level operational means, not universal biological constants.

Source: Benjamin Stucky (2025), *We are the Sensors of Consciousness! A Review and Analysis on How Awakenings During Sleep Influence Dream Recall*, **Nature and Science of Sleep, 17**, 709--729. [DOI](https://doi.org/10.2147/NSS.S506461) · [PubMed/PMC record](https://pubmed.ncbi.nlm.nih.gov/40330584/) · [open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12053782/)

## DREAM database pooled cross-check

Wong et al. (2025) harmonized 20 datasets (505 participants; 2,643 awakenings in the initial database). Their behavioral contingency table contains the **1,550 awakenings with unambiguous stage and experience classifications**. “Experience” is deliberately broad: any recalled conscious experience from sleep, including thoughts or isolated imagery, not only vivid narrative dreams.

| Last scored stage | No experience | Experience without recall | Experience | Total |
|---|---:|---:|---:|---:|
| N1 | 12 (11%) | 1 (1%) | **97 (88%)** | 110 |
| N2 | 308 (36%) | 68 (8%) | **485 (56%)** | 861 |
| N3/NREM3/NREM4 | 26 (41%) | 7 (11%) | **31 (48%)** | 64 |
| REM | 87 (17%) | 12 (2%) | **416 (81%)** | 515 |
| Total | 433 (28%) | 88 (6%) | 1,029 (66%) | 1,550 |

These raw percentages independently reproduce the central pattern `N1 high`, `N2/N3 moderate`, and `REM high`. A mixed-effects analysis accounting for study and participant clustering found REM had higher odds of “Experience” than N2 but lower odds than N1. The paper also summarizes the established literature as about 85% of REM awakenings and 40--60% of NREM awakenings yielding dream reports.

Source: William Wong, Rubén Herzog, Kátia C. Andrade, et al. (2025), *A dream EEG and mentation database*, **Nature Communications, 16**, 7495. [DOI/open article](https://doi.org/10.1038/s41467-025-61945-1) · [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12350935/) · [database DOI](https://doi.org/10.26180/22133105)

## Independent protocol-level checks

| Source and protocol | Content/dream recall by stage | Interpretation |
|---|---|---|
| Siclari et al. (2013): 7 healthy adults, 44 laboratory nights, serial sound awakenings every 15--30 min regardless of stage; broad “last thing going through your mind” prompt | W 96±7%; **N1 77±39%; N2 42±15%; N3 23±15%; REM 82±19%** (participant-wise mean±SD; 741 retained questionings) | Supports high N1/REM and nontrivial N2. The abstract's pooled NREM value is 34%, which is not an N1 estimate. |
| Nielsen (2000): review of 35 REM/NREM studies | REM **81.9±9.0%**; pooled NREM **43.0±20.8%** | Long-standing cross-study benchmark; also emphasizes that broad cognitive activity gives higher NREM rates than strict “dreaming” criteria. |
| Picard-Deland et al. (2023): 20 healthy frequent dream recallers, serial awakenings across stage and time of night | N1 44/58 = **75.8%**; N2 41/60 = **68.3%**; N3 30/53 = **56.6%**; REM 49/56 = **87.5%** | Useful within-protocol check, but participants were selected for at least three recalled dreams/week. |
| Picard-Deland, Nielsen & Carr (2021): pooled Montreal laboratory sample, 403 participants, 650 awakenings; liberal prompt | N1 104/139 = **74.8%**; N2 78/124 = **62.9%**; N3 9/14 = **64.3%**; REM 337/373 = **90.3%** | Large individual sample but heterogeneous naps/nights and some experimental manipulations; N3 denominator is only 14. |

Sources:

- Francesca Siclari, Joshua J. LaRocque, Bradley R. Postle & Giulio Tononi (2013), *Assessing sleep consciousness within subjects using a serial awakening paradigm*, **Frontiers in Psychology, 4**, 542. [DOI/open article](https://doi.org/10.3389/fpsyg.2013.00542)
- Tore A. Nielsen (2000), *A review of mentation in REM and NREM sleep: “Covert” REM sleep as a possible reconciliation of two opposing models*, **Behavioral and Brain Sciences, 23**, 851--866. [DOI](https://doi.org/10.1017/S0140525X0000399X) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/11515145/)
- Claudia Picard-Deland, Karen Konkoly, Rachel Raider, et al. (2023), *The memory sources of dreams: serial awakenings across sleep stages and time of night*, **Sleep, 46**, zsac292. [DOI](https://doi.org/10.1093/sleep/zsac292) · [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10091095/)
- Claudia Picard-Deland, Tore Nielsen & Michelle Carr (2021), *Dreaming of the sleep lab*, **PLOS ONE, 16**, e0257738. [DOI/open article](https://doi.org/10.1371/journal.pone.0257738)

## Why `0.35 / 0.15 / 0.85` is not a supported stage-probability triplet

### REM = 0.85

This is a defensible rounded mean. It is close to Stucky's 83.4%, Nielsen's 81.9%, the DREAM raw 81%, and several targeted cohorts' 82--90%.

### N1 = 0.35

This is far below every broad immediate-awakening aggregate above. N1 is the sleep-onset transition, where hypnagogic imagery and other reportable mentation are common. A value near 35% appears in Siclari et al. as **34% pooled across all NREM stages**, not as the N1 mean; their N1 mean was 77%. Thus 35% may be a conflation of pooled NREM with N1, but the provenance of the proposed number cannot be established without the original citation.

### N2 = 0.15

No authoritative stage-specific mean near 15% was identified for broad immediate content recall. In Siclari et al., the N2 estimate is **42±15%**: 15 is the between-participant SD, not the mean. Stucky's mean is 53.2% and DREAM's raw pooled result is 56%. Again, the proposed 15% could be a stricter heuristic for vivid/story-like dreams, but it should not be presented as the mean probability of a recalled sleep experience.

### Could stricter “canonical dream” criteria justify the lower values?

Possibly lower rates, yes; this exact triplet, no. Rates fall when studies require vivid, perceptual, immersive, or narratively structured content rather than any mental experience. However, there is no single harmonized, stage-specific synthesis establishing `(N1,N2,REM)=(0.35,0.15,0.85)` for that stricter construct. If these values are retained, label them a **modeling heuristic/sensitivity prior**, not an empirical consensus.

## Follow-up: probability of a recalled dream with a clear narrative plot

The more specific quantity requested in the follow-up is

\[
P(\text{recalled coherent narrative}\mid\text{awakening from stage }s).
\]

This is not the same outcome as the broad content-recall probability used above.
An adequate replacement for the model weights would need the same denominator
(all awakenings), the same operational definition of a clear narrative, and
separate estimates for N1, N2, N3, and REM. **No study or meta-analysis meeting
all of those requirements was identified.** Existing work nevertheless shows
that narrative quality is measurable and that REM reports are often more
story-like; the problem is comparability, not an absence of research.

| Evidence | Relevant result | Why it is not a full-stage probability table |
|---|---|---|
| Martin et al. (2020), 198 controlled REM/N2 awakenings in 20 participants | The protocol obtained 146 recalled reports. The final analysis retained 133 reports of at least 30 words (46 REM, 87 N2); within this selected analytic sample, the reported proportions describing elaborate ongoing narrative sequences were **75.09% for REM** and **43.68% for N2**. | The denominator is an already recalled, length-filtered report rather than every awakening; N1 and N3 are absent. Therefore these figures are not `P(narrative report | awakening, stage)`. |
| Nielsen et al. (2001), 380 experimental REM/N2 awakening reports in 24 participants | At least one story constituent occurred in subject-wise means of **44% REM** and **29% N2** reports. Among a selected analysis of participants with constituent recall in both stages, episodic progression occurred in **66% REM** and **43% N2** reports. Effects varied strongly with habitual recall frequency and time/order of awakening. | The two percentages address different narrative thresholds and denominators; the episodic-progression result is conditional on a selected recall-capable subset. N1 and N3 are absent. |
| Nielsen (2017), review of sleep-onset imagery | Any sleep-onset imagery in combined N1/N2 was reported at 90--98%, whereas more complex hallucinatory dramatic episodes spanned **31--76%**. | N1 and N2 are combined, the range is wide and protocol-dependent, and no matching REM/N3 estimates using the same criterion are supplied. |
| Immersive NREM2 study (2026), 1,024 N2 awakenings | The study quantified a continuous *perceptual immersion* component from duration, vividness, perceptual richness, bizarreness, and emotion in recalled N2 experiences. | Immersion is a continuous phenotype rather than a prespecified binary clear-plot event, and only N2 was sampled. |

Sources:

- Joshua M. Martin, Danyal W. Andriano, Natalia B. Mota, et al. (2020), *Structural differences between REM and non-REM dream reports assessed by graph analysis*, **PLOS ONE, 15**, e0228903. [DOI/open article](https://doi.org/10.1371/journal.pone.0228903)
- Tore Nielsen, Don Kuiken, Robert Hoffmann & Alan Moffitt (2001), *REM and NREM sleep mentation differences: A question of story structure?*, **Sleep and Hypnosis, 3**, 9--17. [Open PDF](https://www.dreamscience.ca/en/documents/publications/_2001_Nielsen_Kuiken_S%26H_3_9-17_REM_%26_NREM_mentation.pdf)
- Tore Nielsen (2017), *Microdream neurophenomenology*, **Neuroscience of Consciousness, 2017**, nix001. [DOI](https://doi.org/10.1093/nc/nix001) · [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6007184/)
- Adriana Michalak, Davide Marzoli, Francesco Pietrogiacomi, et al. (2026), *Immersive NREM2 dreaming preserves subjective sleep depth against declining sleep pressure*, **PLOS Biology, 24**, e3003683. [DOI/open article](https://doi.org/10.1371/journal.pbio.3003683)

### Modeling decision after this follow-up

The primary optimization therefore continues to use the comparable broad
content-recall means `(W,N1,N2,N3,REM) = (0,.849,.532,.508,.834)`. They remain
**reportability priors**, not probabilities of a vivid or coherent story. The
Martin REM/N2 narrative figures are retained only as supporting evidence and a
possible future sensitivity analysis; inserting them into the primary model
would mix denominators, omit two stages, and introduce an arbitrary rule for
N1/N3.

## Recommended use in the `g` optimization

Use the 69-study **experience-with-recall** means as fixed external weights:

| Model stage `s` | Fixed primary weight `p_s` | Provenance |
|---|---:|---|
| Wake | 0.000 | Target definition: wake time is not dream time |
| N1 | 0.849 | 69-study mean, experience with recall |
| N2 | 0.532 | 69-study mean, experience with recall |
| N3 | 0.508 | 69-study mean, experience with recall |
| REM | 0.834 | 69-study mean, experience with recall |

For hard GSSC stage labels `S(t)`, compute

\[
\widehat T_{\mathrm{dream}}
=\int_0^T p_{S(t)}\,g\{f(t)\}\,dt.
\]

For regular epochs of duration `Δt`, the implementation is simply

\[
\widehat T_{\mathrm{dream}}
=\Delta t\sum_i p_{S_i}\,g(f_i).
\]

If GSSC supplies well-calibrated stage posteriors `q_s(t)`, an optional uncertainty-aware version is

\[
p_{\mathrm{eff}}(t)=\sum_s q_s(t)p_s,
\qquad
\widehat T_{\mathrm{dream}}
=\int_0^T p_{\mathrm{eff}}(t)\,g\{f(t)\}\,dt.
\]

Do not assume that raw neural-network softmax scores are calibrated. Unless calibration is verified, use hard labels for the primary analysis and reserve posterior weighting for sensitivity analysis.

### Re-optimization and identifiability

Introducing `p_{S(t)}` changes the forward model, so a `g` optimized under the old integral should be **re-optimized on the revised training set**. The literature weights themselves should remain fixed: jointly fitting the overall scale of `p_s` and `g` from the same duration labels creates avoidable scale/identifiability confounding. Lock the complete modeling choice before evaluating the held-out test set.

Suggested transparent sensitivity set:

1. Primary: Stucky 69-study means `(0, .849, .532, .508, .834)`.
2. External cross-check: DREAM raw pooled rates `(0, .88, .56, .48, .81)`.
3. User heuristic: `(0, .35, .15, p_N3, .85)`, with `p_N3` explicitly declared rather than silently omitted.
4. Existing REM-only formulation, if applicable, as a baseline.

The test set should not be used to select among these priors.

## Operational-prior limitation (required manuscript wording)

> The stage weights are literature-derived probabilities of reporting recalled conscious content after an awakening from a given stage, under heterogeneous laboratory and home protocols. They are used here as fixed operational priors. They are not direct estimates of the instantaneous probability of dreaming, the fraction of a stage spent dreaming, or the probability of vivid narrative dreaming. A null report may reflect absent experience, failed encoding, or failed retrieval. Converting these cross-sectional awakening rates into a continuous temporal multiplier is therefore a modeling assumption.

This limitation matters for interpretation: `p_s` should be described as a stage-dependent **reportability/recallability prior**, while `g(f)` supplies the within-stage signal-dependent term. The resulting integral estimates an operational expected dream-duration proxy, not a directly observed physiological duty cycle.
