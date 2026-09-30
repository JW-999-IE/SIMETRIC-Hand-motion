# SIMETRIC — Hand Motion and Needle-Entry Grip Posture

Analysis code, frozen datasets and model outputs for the hand-kinematic and grip-posture
components of the SIMETRIC study (Simulation and Imaging Methods for Eye Tracking and Recording
Intravenous Catheter Insertion).

Companion repositories:
[eye tracking](https://github.com/JW-999-IE/SIMETRIC-Eye-tracking) ·
[surveys](https://github.com/JW-999-IE/SIMETRIC-Surveys-Pre-and-Post-task)

## Study context

Twenty-seven hospital clinicians each performed five ultrasound-guided long peripheral catheter
(LPC) insertions with each of three configurations — catheter-over-needle (CON), accelerated
technique with guidewire (ATG), and modified Seldinger technique (MST) — on a vascular-access
phantom, yielding 414 non-aborted attempts (ATG 141, CON 136, MST 137).

**Terminology.** The parent protocol refers to these devices as midline catheters. At 8 cm
insertable length they fall within the long peripheral catheter category under current
definitions. LPC is used throughout.

**Configuration identity.** `DEVICE1 = MST` (modified Seldinger technique), `DEVICE2 = ATG`
(accelerated technique with guidewire), `DEVICE3 = CON` (catheter-over-needle). A device-identity
review of 18 MST-labelled attempts proposed two reassignments (P6_T11 → ATG, P19_T03 → CON); both
were rejected on adjudication against the session Seldinger phase records, and the original
workbook identities are retained for all 414 attempts.

## Hand motion

### Capture and processing

Two synchronised GoPro HERO cameras captured left and right hand views. GoPro video is the
study-primary source; CAE LearningSpace HAND-camera recordings were used only for prespecified
recovery and sensitivity analyses. Hand landmarks were extracted frame-by-frame with Google
MediaPipe Hands v0.10.14, with the wrist landmark used for trajectory and scalar measures.

Cameras were monocular and not geometrically fused, so **all coordinates are normalised
two-dimensional image-space measures, not calibrated physical distances or velocities.** Path
length, displacement and speed should be read accordingly.

### Analysis populations

| Set | n | Note |
|---|---|---|
| Canonical attempt-hand universe | 828 | 414 attempts × 2 hands |
| GoPro-primary scalar | 251 | 92 left-view, 159 right-view, 20 participants |
| Scalar with CAE sensitivity additions | 256 | 5 non-overlapping CAE additions |
| GoPro-primary trajectory | 244 | |
| Trajectory with CAE additions | 249 | |

The scalar set represents 30% of the canonical universe and excludes seven of 27 clinicians. Loss
arose because the operator's head or hands intermittently occluded the camera view, preventing
reliable landmark extraction; the left-view camera was more often obstructed, which accounts for
the asymmetry between views. Hand motion is the least complete domain in the study.

### Prespecified outcomes

Normalised path length, path efficiency, mean normalised speed, log dimensionless jerk, and
stillness fraction over observed tracking. Secondary: normalised displacement, peak normalised
speed, velocity-peak count, SPARC, stillness-episode count, longest stillness duration.

Models are Gaussian participant random-intercept mixed models fitted by maximum likelihood, as a
function of configuration, prior LPC experience, hand and round, with MST, no prior experience and left view as reference
levels. Path length and mean speed use log1p transformation; path efficiency and stillness
fraction use logit; log dimensionless jerk is modelled on its native scale. Transformed outcomes
are standardised using the frozen GoPro-primary mean and SD. Benjamini-Hochberg correction is
applied within outcome families.

### Headline results (GoPro-primary, SD units)

| Outcome | Omnibus | ATG vs MST | CON vs MST | ATG vs CON |
|---|---|---|---|---|
| Normalised path length | χ²(2)=135.61; P<0.001 | −0.887 (−1.084, −0.691) | −1.055 (−1.244, −0.867) | +0.168 (−0.025, 0.362); q=0.152 |
| Mean normalised speed | χ²(2)=40.25; P<0.001 | −0.614 (−0.840, −0.388) | −0.623 (−0.840, −0.406) | +0.009 (−0.213, 0.232); q=0.933 |
| Stillness fraction | χ²(2)=15.93; P<0.001 | +0.340 (0.068, 0.612) | +0.528 (0.266, 0.790) | −0.188 (−0.458, 0.082); q=0.236 |
| Path efficiency | χ²(2)=7.77; P=0.021 | +0.240 (−0.039, 0.519); q=0.152 | +0.378 (0.110, 0.646); q=0.014 | −0.137 (−0.413, 0.138); q=0.409 |
| Log dimensionless jerk | χ²(2)=2.49; P=0.288 | not significant | not significant | not significant |

ATG and CON do not differ on any of the five outcomes. The kinematic separation is between MST
and the two single-unit configurations. Adjusting for attempt duration attenuates the path-length
effects by 39.4% (ATG vs MST) and 44.0% (CON vs MST) but both remain strongly FDR-significant, so
duration is one mechanism rather than the whole explanation.

**Reporting note.** Figures reported in the manuscripts are from the **GoPro-primary** package
throughout. The integrated GoPro+CAE package is a sensitivity analysis and gives slightly
different estimates (e.g. path length CON vs MST −1.085 rather than −1.055). Do not mix the two.

## Grip posture

### Classification

Grip was classified manually from video at the exact needle-entry moment into four categories:
external precision grip (EPG, shaft external to the palm), internal precision grip (IPG, shaft
across or within the palm), thumb curl (TC), and other (grip visible but not defensibly
assignable). The shaft-palm relationship is the principal EPG/IPG discriminator; thumb position
alone is not sufficient for TC. Automated candidate classifiers did not meet the prespecified
validation threshold and were not used as final labels.

Classification proceeded through successive frozen releases. Freeze v1.6 resolved 324 of 414
attempts with 90 unresolved for want of a located video source; those were subsequently recovered
by targeted video localisation. **Freeze v1.8.2 is the authoritative dataset used here:** 414 of
414 resolved, none unresolved, SHA256-manifested.

### Distribution (adjudicated configuration identities)

| Grip | ATG (n=141) | CON (n=136) | MST (n=137) | Total |
|---|---|---|---|---|
| IPG | 56 (39.7%) | 76 (55.9%) | 59 (43.1%) | 191 (46.1%) |
| EPG | 28 (19.9%) | 24 (17.6%) | 51 (37.2%) | 103 (24.9%) |
| TC | 39 (27.7%) | 34 (25.0%) | 17 (12.4%) | 90 (21.7%) |
| Other | 18 (12.8%) | 2 (1.5%) | 10 (7.3%) | 30 (7.2%) |

### Model and results

Participant-clustered multinomial (nominal) GEE with a global odds-ratio working structure, IPG
as reference grip, MST as reference configuration, no prior experience as reference stratum;
covariates configuration, prior LPC experience and within-configuration repetition. Benjamini-Hochberg
correction across the six configuration-by-category contrasts and, separately, across the six
experience-by-category contrasts.

Configuration χ²(6) = 52.90, P < 0.001. Prior LPC experience χ²(6) = 15.85, P = 0.015. Repetition
χ²(3) = 1.18, P = 0.759.

| Contrast | Grip vs IPG | RRR (95% CI) | q |
|---|---|---|---|
| CON vs MST | EPG | 0.36 (0.20–0.67) | **0.006** |
| ATG vs MST | TC | 2.66 (1.31–5.42) | **0.021** |
| CON vs MST | Other | 0.18 (0.04–0.82) | 0.054 |
| ATG vs MST | Other | 2.50 (1.00–6.27) | 0.077 |
| ATG vs MST | EPG | 0.55 (0.29–1.07) | 0.092 |
| CON vs MST | TC | 1.64 (0.80–3.37) | 0.175 |
| Higher vs no prior experience | Other | 3.31 (1.41–7.76) | 0.035 |

Both principal contrasts survive restriction to the first five observed repetitions per
configuration (RRR 2.49 and 0.36, q = 0.010 each) and exclusion of the residual Other category
entirely (q = 0.006 each), so the configuration effect does not depend on unclassifiable
postures.

The prior-experience association is **not robust to individual clinicians**: in leave-one-clinician-out
refits it held in 24 of 26 convergent refits, but removing either of two higher-experience clinicians
abolished it (RRR 2.34, P = 0.116; RRR 1.95, P = 0.152) while removing either of two others
strengthened it (RRR 5.17 and 4.02). With five higher-experience clinicians contributing, it is reported as
exploratory only.

Inter-rater agreement on the classifiable validation comparison was 55.6% (69/124), Cohen's
κ = 0.344, Gwet's AC1 = 0.432. The grip endpoint is exploratory.

## Repository structure

```
├── README.md
├── scripts/
│   ├── pipeline/          # CAE sidecar pipeline, extraction through modelling
│   ├── refit_grip.py      # multinomial GEE grip refit on freeze v1.8.2
│   └── grip_spec_and_expertise.py   # specification comparison and expertise omnibus
├── data/
│   └── grip_freeze_v1_8_2/          # authoritative grip dataset, SHA256-manifested
├── results/
│   ├── scalar_gee/                  # scalar hand-motion GEE outputs
│   ├── hand_motion_statistics/      # LMM coefficients, omnibus tests, contrasts
│   ├── manuscript_package_gopro/    # GoPro-primary tables as reported
│   ├── manuscript_package_integrated/  # integrated GoPro+CAE sensitivity
│   ├── publication_qc/              # sample-flow and verification tables
│   └── grip/                        # grip contrasts, transition matrix, LOPO
└── figures/
```

## Data availability

Raw video recordings contain human participant data and are not deposited. Derived landmark
trajectories, frozen analysis datasets and all model outputs are included.

## Ethics

Approved by the Clinical Research Ethics Committee at the University of Galway (C.A. 3392) before
data collection began. Written informed consent was obtained from all participants.

## Requirements

```
Python >= 3.9
numpy, pandas, scipy, statsmodels
mediapipe == 0.10.14     # landmark extraction only
opencv-python            # video handling
matplotlib, seaborn      # figures
```

Reported analyses were run under Python 3.14.5, statsmodels 0.14.6, scipy 1.17.1, numpy 2.4.6,
pandas 3.0.3.

## Citation

Manuscript reference to be added on acceptance.
