from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Freeze the pre-specified SIMETRIC hand-motion analysis populations "
            "before any outcome/device-effect modelling."
        )
    )
    ap.add_argument("--master", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    master_path = Path(args.master)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(master_path, dtype=str).fillna("")

    required = {
        "analysis_gopro_ge_60",
        "analysis_gopro_ge_80",
        "analysis_allsource_ge_60",
        "analysis_allsource_ge_50",
        "participant",
        "hand",
        "device",
        "motion_source",
        "experience_stratum",
        "attempt_id",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError("Missing required columns: " + ", ".join(sorted(missing)))

    df["analysis_main"] = as_bool(df["analysis_gopro_ge_60"])
    df["analysis_highqc_sensitivity"] = as_bool(df["analysis_gopro_ge_80"])
    df["analysis_recovery_sensitivity"] = as_bool(df["analysis_allsource_ge_60"])
    df["analysis_permissive_sensitivity"] = as_bool(df["analysis_allsource_ge_50"])

    sets = {
        "main_gopro_ge60": "analysis_main",
        "highqc_gopro_ge80": "analysis_highqc_sensitivity",
        "recovery_allsource_ge60": "analysis_recovery_sensitivity",
        "permissive_allsource_ge50": "analysis_permissive_sensitivity",
    }

    summary_rows = []
    for name, col in sets.items():
        sub = df[df[col]]
        summary_rows.append({
            "analysis_set": name,
            "rows": len(sub),
            "participants": sub["participant"].nunique(),
            "participant_hand_groups": len(sub[["participant","hand"]].drop_duplicates()),
            "left_rows": int((sub["hand"] == "left").sum()),
            "right_rows": int((sub["hand"] == "right").sum()),
            "MST_rows": int((sub["device"] == "MST").sum()),
            "ATG_rows": int((sub["device"] == "ATG").sum()),
            "CON_rows": int((sub["device"] == "CON").sum()),
            "gopro_left_rows": int((sub["motion_source"] == "gopro_left").sum()),
            "gopro_right_rows": int((sub["motion_source"] == "gopro_right").sum()),
            "cae_rows": int((sub["motion_source"] == "cae_hand").sum()),
        })

    summary = pd.DataFrame(summary_rows)

    manifest = {
        "population_freeze_status": "FROZEN_BEFORE_OUTCOME_MODELLING",
        "main_analysis": {
            "definition": "GoPro only; qc_status PASS; tracking_coverage >= 0.60; observed_span_fraction >= 0.60",
            "flag": "analysis_main",
            "rationale": (
                "Pre-outcome threshold selected from tracking quality and cohort representation. "
                "GoPro >=0.60 retains the same 24 participants as >=0.50 and 42/54 participant-hand "
                "groups versus 43/54 at >=0.50, while >=0.70 drops to 22 participants and 38 groups."
            ),
        },
        "high_quality_sensitivity": {
            "definition": "GoPro only; tracking_coverage and observed_span_fraction >=0.80",
            "flag": "analysis_highqc_sensitivity",
        },
        "recovery_sensitivity": {
            "definition": (
                "All validated sources; tracking_coverage and observed_span_fraction >=0.60; "
                "motion_source must remain in the statistical model."
            ),
            "flag": "analysis_recovery_sensitivity",
        },
        "permissive_sensitivity": {
            "definition": (
                "All validated sources; tracking_coverage and observed_span_fraction >=0.50; "
                "used only as a broader robustness analysis."
            ),
            "flag": "analysis_permissive_sensitivity",
        },
        "scalar_model_covariates": [
            "device",
            "experience_stratum",
            "device_repetition_c",
            "chronological_attempt_sequence_c",
        ],
        "hand_rule": "Model left and right hands separately.",
        "repeated_measure_rule": "Cluster repeated attempts within participant.",
        "round_rule": (
            "Do not include round together with device_repetition because they are nearly redundant; "
            "inspect the isolated disagreement separately and use device_repetition derived from the "
            "detailed workbook device instance as the repetition covariate."
        ),
        "grip_rule": "Screening only; exclude from inference until manual validation.",
        "tremor_rule": "Exploratory only; tracking/camera-jitter sensitive.",
        "trajectory_rule": (
            "Do not reduce the gap-aware 100-point trajectory to the same scalar GEE. "
            "Analyse trajectory separately with missing points preserved."
        ),
    }

    out_master = outdir / "statistical_master_attempt_hand_828_ANALYSIS_FROZEN.csv"
    out_summary = outdir / "analysis_population_freeze_summary.csv"
    out_manifest = outdir / "analysis_population_freeze_manifest.json"

    df.to_csv(out_master, index=False)
    summary.to_csv(out_summary, index=False)
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=== ANALYSIS POPULATIONS FROZEN ===")
    print(summary.to_string(index=False))
    print()
    print("Wrote:", out_master)
    print("Wrote:", out_summary)
    print("Wrote:", out_manifest)
    print()
    print("Main analysis is now locked BEFORE device-effect modelling:")
    print("  GoPro >=60% tracking coverage AND >=60% observed-span coverage.")
    print("Do not change this threshold based on subsequent p-values.")


if __name__ == "__main__":
    main()
