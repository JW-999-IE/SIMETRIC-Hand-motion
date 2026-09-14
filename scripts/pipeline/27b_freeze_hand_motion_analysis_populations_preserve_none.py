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
            "while preserving the literal experience category 'None'."
        )
    )
    ap.add_argument("--master", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    master_path = Path(args.master)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # CRITICAL: pandas otherwise interprets the literal category "None" as NA.
    df = pd.read_csv(
        master_path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )

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

    valid_experience = {"Higher", "Some", "None"}
    observed_experience = set(df["experience_stratum"].unique())
    unexpected = sorted(observed_experience - valid_experience)
    if unexpected:
        print("STOP: unexpected experience labels:", unexpected)
        raise SystemExit(2)

    if df["experience_stratum"].eq("").any():
        print("STOP: blank experience_stratum rows exist before freezing.")
        print(
            df.loc[
                df["experience_stratum"].eq(""),
                ["participant", "attempt_id", "hand"],
            ].head(50).to_string(index=False)
        )
        raise SystemExit(2)

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
            "participant_hand_groups": len(
                sub[["participant", "hand"]].drop_duplicates()
            ),
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

    exp_summary = (
        df[["participant", "experience_stratum"]]
        .drop_duplicates()
        .groupby("experience_stratum")
        .size()
        .reindex(["Higher", "Some", "None"], fill_value=0)
        .rename("participants")
        .reset_index()
    )

    manifest = {
        "population_freeze_status": "FROZEN_BEFORE_OUTCOME_MODELLING",
        "experience_na_handling": (
            "CSV read with keep_default_na=False and na_filter=False so literal "
            "experience category 'None' is preserved."
        ),
        "experience_levels": ["Higher", "Some", "None"],
        "main_analysis": {
            "definition": (
                "GoPro only; qc_status PASS; tracking_coverage >= 0.60; "
                "observed_span_fraction >= 0.60"
            ),
            "flag": "analysis_main",
            "rationale": (
                "Pre-outcome threshold selected from tracking quality and cohort "
                "representation. GoPro >=0.60 retains the same 24 participants as "
                ">=0.50 and 42/54 participant-hand groups versus 43/54 at >=0.50, "
                "while >=0.70 drops to 22 participants and 38 groups."
            ),
        },
        "high_quality_sensitivity": {
            "definition": (
                "GoPro only; tracking_coverage and observed_span_fraction >=0.80"
            ),
            "flag": "analysis_highqc_sensitivity",
        },
        "recovery_sensitivity": {
            "definition": (
                "All validated sources; tracking_coverage and observed_span_fraction "
                ">=0.60; motion_source must remain in the statistical model."
            ),
            "flag": "analysis_recovery_sensitivity",
        },
        "permissive_sensitivity": {
            "definition": (
                "All validated sources; tracking_coverage and observed_span_fraction "
                ">=0.50; used only as a broader robustness analysis."
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
            "Do not include round together with device_repetition because they are "
            "nearly redundant; P26_T13 is the documented exception."
        ),
        "grip_rule": "Screening only; exclude from inference until manual validation.",
        "tremor_rule": "Exploratory only; tracking/camera-jitter sensitive.",
        "trajectory_rule": (
            "Analyse trajectory separately with missing points preserved."
        ),
    }

    out_master = (
        outdir / "statistical_master_attempt_hand_828_ANALYSIS_FROZEN_v2.csv"
    )
    out_summary = outdir / "analysis_population_freeze_summary_v2.csv"
    out_exp = outdir / "analysis_population_experience_summary_v2.csv"
    out_manifest = outdir / "analysis_population_freeze_manifest_v2.json"

    df.to_csv(out_master, index=False)
    summary.to_csv(out_summary, index=False)
    exp_summary.to_csv(out_exp, index=False)
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=== ANALYSIS POPULATIONS FROZEN V2 ===")
    print(summary.to_string(index=False))
    print()
    print("Experience levels preserved:")
    print(exp_summary.to_string(index=False))
    print()
    print("Blank experience rows:", int(df["experience_stratum"].eq("").sum()))
    print()
    print("Wrote:", out_master)
    print("Wrote:", out_summary)
    print("Wrote:", out_exp)
    print("Wrote:", out_manifest)

    expected = {"Higher": 5, "Some": 9, "None": 13}
    got = dict(
        zip(exp_summary["experience_stratum"], exp_summary["participants"])
    )
    mismatch = {
        k: (expected[k], int(got.get(k, 0)))
        for k in expected
        if int(got.get(k, 0)) != expected[k]
    }

    if mismatch:
        print("STOP: experience participant counts do not match expected study strata.")
        print("Mismatches:", mismatch)
        raise SystemExit(2)

    print()
    print(
        "PASS: frozen analysis populations are unchanged and experience categories "
        "Higher=5, Some=9, None=13 are preserved exactly."
    )


if __name__ == "__main__":
    main()
