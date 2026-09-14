SIMETRIC FINAL NON-GRIP HAND-MOTION STATISTICS

STATUS
======
COMPLETE if this run reaches FINAL PASS.

PRIMARY DATA
============
251 GoPro-primary attempt-hands.
20 participants.

Primary participant expertise:
    Expert: 3
    Intermediate: 6
    Novice: 11

Expertise findings are secondary/exploratory because only three Expert
participants are represented in the GoPro-primary hand-motion set.

SENSITIVITY DATA
================
256 scalar attempt-hands.
Five CAE scalar additions.
GoPro remains primary.

MODEL
=====
Random-intercept linear mixed model fitted by direct ML profile likelihood:

    transformed z outcome
        ~ authoritative_device
        + expertise_group
        + hand
        + round
        + (1 | participant)

Secondary order model also includes attempt_sequence.

Reference levels:
    Device: MST
    Expertise: Novice
    Hand: left

MAIN OUTCOMES
=============
path_length_norm
path_efficiency
mean_speed_norm_s
log_dimensionless_jerk
stillness_fraction_observed_tracking

SECONDARY OUTCOMES
==================
displacement_norm
peak_speed_norm_s
velocity_peak_count
sparc
stillness_episode_count
longest_stillness_sec

NOT IN MAIN INFERENCE
=====================
path_efficiency_raw:
    replaced analytically by QC-controlled path_efficiency

Acceleration derivatives:
    descriptive only

tremor_* / hf_motion_*:
    excluded from main inference; exploratory high-frequency motion only

grip variables:
    excluded; grip analysis is being completed in the separate branch

MULTIPLICITY
============
Benjamini-Hochberg FDR is applied within dataset/outcome-tier/test family.

TRAJECTORY
==========
100-point speed profiles are summarised descriptively by device and hand after
first aggregating within participant.

No pointwise 100-test significance maps are created.

x/y curves are not pooled for inferential device comparison because the
cameras are uncalibrated non-fused 2D views.

UNITS
=====
Normalised path/kinematic variables remain image-space/normalised measures,
not physical millimetres.

NEXT
====
Review the actual root210 statistical results.

Then create the final hand-motion manuscript tables/figures and write the
hand-motion Results/Methods text. Grip results remain branch-owned until that
workstream is frozen.
