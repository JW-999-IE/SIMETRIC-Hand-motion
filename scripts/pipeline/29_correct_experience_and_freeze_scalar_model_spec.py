from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


OUTCOMES = [
    "mean_accel_norm_s2",
    "rms_accel_norm_s2",
    "peak_accel_norm_s2",
    "sparc",
    "log_dimensionless_jerk",
    "velocity_peak_count",
    "path_length_norm",
    "path_efficiency",
    "stillness_fraction",
    "first_sustained_hand_movement_delay_sec",
]


def to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Corrected pre-model validation and model-specification freeze. "
            "Reads CSV with keep_default_na=False so the legitimate experience "
            "category 'None' is never converted to missing."
        )
    )
    ap.add_argument("--master", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Critical: preserve literal string "None".
    df = pd.read_csv(args.master, dtype=str, keep_default_na=False)

    required = {
        "analysis_main",
        "analysis_highqc_sensitivity",
        "analysis_recovery_sensitivity",
        "participant",
        "hand",
        "device",
        "experience_stratum",
        "device_repetition_c",
        "chronological_attempt_sequence_c",
        "motion_source",
        "observed_continuous_tracking_sec",
    } | set(OUTCOMES)

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError("Missing required columns: " + ", ".join(sorted(missing)))

    valid_experience = {"Higher", "Some", "None"}
    observed_experience = set(df["experience_stratum"].unique())
    bad_experience = sorted(observed_experience - valid_experience)
    if bad_experience:
        print("STOP: unexpected experience labels:", bad_experience)
        raise SystemExit(2)

    if df["experience_stratum"].eq("").any():
        print("STOP: blank experience_stratum rows remain.")
        print(
            df.loc[df["experience_stratum"].eq(""),
                   ["participant", "attempt_id", "hand"]]
            .head(50).to_string(index=False)
        )
        raise SystemExit(2)

    main = df[to_bool(df["analysis_main"])].copy()

    # Numeric checks required by planned models.
    numeric_cols = [
        "device_repetition_c",
        "chronological_attempt_sequence_c",
        "observed_continuous_tracking_sec",
    ] + OUTCOMES
    for c in numeric_cols:
        main[c + "_num"] = pd.to_numeric(main[c], errors="coerce")

    # Experience balance.
    exp_participants = (
        main[["participant", "experience_stratum"]]
        .drop_duplicates()
        .groupby("experience_stratum")
        .size()
        .reindex(["Higher", "Some", "None"], fill_value=0)
        .rename("participants")
        .reset_index()
    )
    exp_rows = (
        main.groupby("experience_stratum")
        .size()
        .reindex(["Higher", "Some", "None"], fill_value=0)
        .rename("rows")
        .reset_index()
    )
    exp_summary = exp_participants.merge(exp_rows, on="experience_stratum")
    exp_path = outdir / "experience_balance_corrected.csv"
    exp_summary.to_csv(exp_path, index=False)

    dev_exp = (
        main.groupby(["hand", "device", "experience_stratum"], dropna=False)
        .agg(
            rows=("attempt_id", "size"),
            participants=("participant", "nunique"),
        )
        .reset_index()
    )
    dev_exp_path = outdir / "device_experience_balance_corrected.csv"
    dev_exp.to_csv(dev_exp_path, index=False)

    # Hand-specific estimability check for experience and device.
    estimability_rows = []
    for hand in ["left", "right"]:
        h = main[main["hand"].eq(hand)]
        row = {
            "hand": hand,
            "rows": len(h),
            "participants": h["participant"].nunique(),
            "device_levels": h["device"].nunique(),
            "experience_levels": h["experience_stratum"].nunique(),
            "Higher_participants": h.loc[h["experience_stratum"].eq("Higher"), "participant"].nunique(),
            "Some_participants": h.loc[h["experience_stratum"].eq("Some"), "participant"].nunique(),
            "None_participants": h.loc[h["experience_stratum"].eq("None"), "participant"].nunique(),
            "MST_participants": h.loc[h["device"].eq("MST"), "participant"].nunique(),
            "ATG_participants": h.loc[h["device"].eq("ATG"), "participant"].nunique(),
            "CON_participants": h.loc[h["device"].eq("CON"), "participant"].nunique(),
        }
        estimability_rows.append(row)

    estimability = pd.DataFrame(estimability_rows)
    estimability_path = outdir / "model_estimability_experience_corrected.csv"
    estimability.to_csv(estimability_path, index=False)

    # Exposure validity for negative-binomial peak-count model.
    exposure = main["observed_continuous_tracking_sec_num"]
    bad_exposure = int((~np.isfinite(exposure) | (exposure <= 0)).sum())

    # Outcome availability by hand.
    availability_rows = []
    for hand in ["left", "right"]:
        h = main[main["hand"].eq(hand)]
        for outcome in OUTCOMES:
            x = pd.to_numeric(h[outcome], errors="coerce")
            availability_rows.append({
                "hand": hand,
                "outcome": outcome,
                "rows_in_hand": len(h),
                "nonmissing_n": int(x.notna().sum()),
                "participants_nonmissing": h.loc[x.notna(), "participant"].nunique(),
            })
    availability = pd.DataFrame(availability_rows)
    availability_path = outdir / "model_outcome_availability_final.csv"
    availability.to_csv(availability_path, index=False)

    # Pre-specified model plan, frozen before device-effect modelling.
    model_plan = {
        "freeze_status": "FROZEN_BEFORE_DEVICE_EFFECT_MODELLING",
        "analysis_population": {
            "main": "analysis_main == True (GoPro >=60% tracking and observed-span coverage)",
            "high_qc_sensitivity": "analysis_highqc_sensitivity == True (GoPro >=80%)",
            "recovery_sensitivity": (
                "analysis_recovery_sensitivity == True (all sources >=60%); "
                "include motion_source as an additional covariate"
            ),
        },
        "stratification": "Fit left and right hands separately.",
        "cluster": "participant",
        "working_correlation": "exchangeable",
        "base_covariates": [
            "C(device, Treatment(reference='CON'))",
            "C(experience_stratum, Treatment(reference='None'))",
            "device_repetition_c",
            "chronological_attempt_sequence_c",
        ],
        "temporal_rule": (
            "Use device_repetition_c and chronological_attempt_sequence_c. "
            "Do not also include round because round and device repetition are 99.8% concordant "
            "and P26_T13 is the documented exception (round 3 vs device repetition 4)."
        ),
        "outcomes": {
            "mean_accel_norm_s2": {
                "family": "Gaussian GEE",
                "transform": "log",
                "effect_scale": "ratio of geometric means",
                "raw_scale_sensitivity": True,
            },
            "rms_accel_norm_s2": {
                "family": "Gaussian GEE",
                "transform": "log",
                "effect_scale": "ratio of geometric means",
                "raw_scale_sensitivity": True,
            },
            "peak_accel_norm_s2": {
                "family": "Gaussian GEE",
                "transform": "log",
                "effect_scale": "ratio of geometric means",
                "raw_scale_sensitivity": True,
            },
            "sparc": {
                "family": "Gaussian GEE",
                "transform": "none",
                "effect_scale": "mean difference",
            },
            "log_dimensionless_jerk": {
                "family": "Gaussian GEE",
                "transform": "none",
                "effect_scale": "mean difference",
            },
            "velocity_peak_count": {
                "family": "Negative-binomial GEE",
                "transform": "none",
                "offset": "log(observed_continuous_tracking_sec)",
                "effect_scale": "rate ratio",
                "poisson_sensitivity": True,
            },
            "path_length_norm": {
                "family": "Gaussian GEE",
                "transform": "log",
                "effect_scale": "ratio of geometric means",
                "raw_scale_sensitivity": True,
            },
            "path_efficiency": {
                "family": "Gaussian GEE with robust SE",
                "transform": "none",
                "effect_scale": "mean difference",
                "note": (
                    "Only nonmissing values are analysed; by construction these are restricted "
                    "to the stricter path-efficiency QC subset."
                ),
                "logit_sensitivity": True,
            },
            "stillness_fraction": {
                "family": "Gaussian GEE with robust SE",
                "transform": "none",
                "effect_scale": "mean difference",
                "logit_nonboundary_sensitivity": True,
            },
            "first_sustained_hand_movement_delay_sec": {
                "family": "Gaussian GEE",
                "transform": "log1p",
                "effect_scale": "ratio-like effect on 1+delay",
                "raw_scale_sensitivity": True,
                "interpretation": "not insertion-specific",
            },
        },
        "device_testing": {
            "omnibus": (
                "For each outcome×hand model, test the 2-df joint device effect "
                "(MST and ATG coefficients relative to CON)."
            ),
            "global_multiplicity": (
                "Apply Benjamini-Hochberg FDR across all outcome×hand omnibus device tests."
            ),
            "pairwise": (
                "Report MST vs CON, ATG vs CON, and MST vs ATG with 95% CI. "
                "Apply Holm adjustment to the three pairwise device contrasts within each outcome×hand."
            ),
            "interpretation_rule": (
                "Emphasize effect estimates and 95% CIs; treat pairwise findings as confirmatory "
                "only when the outcome×hand omnibus device test survives the pre-specified FDR control."
            ),
        },
        "grip": "Exclude from inference pending manual validation.",
        "tremor": "Exploratory only; no confirmatory device-effect inference.",
        "trajectory": "Separate longitudinal/functional analysis; not part of these scalar GEE models.",
    }

    plan_path = outdir / "scalar_model_specification_FROZEN.json"
    plan_path.write_text(json.dumps(model_plan, indent=2), encoding="utf-8")

    print("=== CORRECTED EXPERIENCE / MODEL-SPECIFICATION AUDIT ===")
    print("Main rows:", len(main))
    print()
    print("Experience balance:")
    print(exp_summary.to_string(index=False))
    print()
    print("Hand-specific estimability:")
    print(estimability.to_string(index=False))
    print()
    print("Device × experience balance:")
    print(dev_exp.to_string(index=False))
    print()
    print("Invalid/nonpositive velocity-peak exposure rows:", bad_exposure)
    print()
    print("Outcome availability:")
    print(availability.to_string(index=False))
    print()
    print("Wrote:", exp_path)
    print("Wrote:", dev_exp_path)
    print("Wrote:", estimability_path)
    print("Wrote:", availability_path)
    print("Wrote:", plan_path)

    blockers = bad_exposure
    if estimability["experience_levels"].min() < 3:
        blockers += 1
    if estimability["device_levels"].min() < 3:
        blockers += 1

    print()
    print("Pre-model blockers:", blockers)
    if blockers:
        print("STOP before inferential modelling.")
    else:
        print(
            "PASS: literal experience category 'None' is preserved, all three experience "
            "and device levels are estimable within both hands, and the scalar model "
            "specification is frozen before device-effect modelling."
        )


if __name__ == "__main__":
    main()
