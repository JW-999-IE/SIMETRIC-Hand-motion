from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CORE_OUTCOMES = [
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
            "Final analysis-readiness audit for the 828-row SIMETRIC statistical master. "
            "Checks participant-hand representation, zero-eligible strata, outcome availability, "
            "device/hand/source balance, and primary-vs-sensitivity coverage."
        )
    )
    ap.add_argument("--master", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    master = pd.read_csv(args.master, dtype=str).fillna("")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    primary = to_bool(master["analysis_set_primary_gopro"])
    sensitivity = to_bool(master["analysis_set_all_sources_sensitivity"])

    # Participant-hand representation.
    ph = (
        master.groupby(["participant", "hand"], dropna=False)
        .agg(
            attempts_expected=("attempt_id", "size"),
            mapping_unavailable=("mapping_status", lambda s: int((s == "unavailable").sum())),
            metric_rows=("_metric_merge", lambda s: int((s == "both").sum())),
            qc_pass=("qc_status", lambda s: int((s == "PASS").sum())),
            qc_warn=("qc_status", lambda s: int((s == "WARN_LOW_TRACK").sum())),
        )
        .reset_index()
    )

    p_counts = (
        master.loc[primary]
        .groupby(["participant", "hand"])
        .size()
        .rename("primary_rows")
        .reset_index()
    )
    s_counts = (
        master.loc[sensitivity]
        .groupby(["participant", "hand"])
        .size()
        .rename("sensitivity_rows")
        .reset_index()
    )

    ph = ph.merge(p_counts, on=["participant", "hand"], how="left")
    ph = ph.merge(s_counts, on=["participant", "hand"], how="left")
    ph[["primary_rows", "sensitivity_rows"]] = (
        ph[["primary_rows", "sensitivity_rows"]].fillna(0).astype(int)
    )
    ph["zero_primary_rows"] = ph["primary_rows"].eq(0)
    ph["zero_sensitivity_rows"] = ph["sensitivity_rows"].eq(0)

    ph_path = outdir / "analysis_readiness_participant_hand.csv"
    ph.to_csv(ph_path, index=False)

    # Outcome availability by analysis set.
    outcome_rows = []
    for name, mask in [
        ("primary_gopro", primary),
        ("all_sources_sensitivity", sensitivity),
    ]:
        subset = master.loc[mask].copy()
        for outcome in CORE_OUTCOMES:
            if outcome not in subset.columns:
                continue
            vals = pd.to_numeric(subset[outcome], errors="coerce")
            outcome_rows.append({
                "analysis_set": name,
                "outcome": outcome,
                "eligible_rows": len(subset),
                "nonmissing_outcome_rows": int(vals.notna().sum()),
                "missing_outcome_rows": int(vals.isna().sum()),
                "participants_with_nonmissing": int(
                    subset.loc[vals.notna(), "participant"].nunique()
                ),
                "participant_hand_groups_with_nonmissing": int(
                    subset.loc[vals.notna(), ["participant", "hand"]]
                    .drop_duplicates().shape[0]
                ),
            })
    outcome = pd.DataFrame(outcome_rows)
    outcome_path = outdir / "analysis_readiness_outcomes.csv"
    outcome.to_csv(outcome_path, index=False)

    # Balance tables.
    balances = []
    for name, mask in [
        ("primary_gopro", primary),
        ("all_sources_sensitivity", sensitivity),
    ]:
        sub = master.loc[mask]
        for dimension in ["hand", "device", "motion_source", "workbook_device_family"]:
            if dimension not in sub.columns:
                continue
            counts = (
                sub.groupby(dimension, dropna=False)
                .size()
                .reset_index(name="rows")
            )
            counts.insert(0, "analysis_set", name)
            counts.insert(1, "dimension", dimension)
            counts = counts.rename(columns={dimension: "level"})
            balances.append(counts[["analysis_set", "dimension", "level", "rows"]])

    balance = pd.concat(balances, ignore_index=True) if balances else pd.DataFrame()
    balance_path = outdir / "analysis_readiness_balance.csv"
    balance.to_csv(balance_path, index=False)

    # Participant representation in each set.
    participant = (
        master[["participant"]]
        .drop_duplicates()
        .sort_values("participant")
        .copy()
    )
    p_part = (
        master.loc[primary]
        .groupby("participant")
        .size()
        .rename("primary_rows")
    )
    s_part = (
        master.loc[sensitivity]
        .groupby("participant")
        .size()
        .rename("sensitivity_rows")
    )
    participant = participant.merge(p_part, on="participant", how="left")
    participant = participant.merge(s_part, on="participant", how="left")
    participant[["primary_rows", "sensitivity_rows"]] = (
        participant[["primary_rows", "sensitivity_rows"]].fillna(0).astype(int)
    )
    participant["represented_primary"] = participant["primary_rows"] > 0
    participant["represented_sensitivity"] = participant["sensitivity_rows"] > 0

    part_path = outdir / "analysis_readiness_participants.csv"
    participant.to_csv(part_path, index=False)

    # Console summary.
    print("=== FINAL ANALYSIS READINESS AUDIT ===")
    print("Master rows:", len(master))
    print("Participants:", master["participant"].nunique())
    print("Participant-hand groups:", ph.shape[0])
    print("Primary GoPro rows:", int(primary.sum()))
    print("Sensitivity rows:", int(sensitivity.sum()))
    print()
    print("Participants represented in primary:",
          int(participant["represented_primary"].sum()), "/", len(participant))
    print("Participants represented in sensitivity:",
          int(participant["represented_sensitivity"].sum()), "/", len(participant))
    print("Participant-hand groups with ZERO primary rows:",
          int(ph["zero_primary_rows"].sum()), "/", len(ph))
    print("Participant-hand groups with ZERO sensitivity rows:",
          int(ph["zero_sensitivity_rows"].sum()), "/", len(ph))

    zero_p = ph[ph["zero_primary_rows"]]
    if len(zero_p):
        print("\n=== ZERO PRIMARY PARTICIPANT-HAND GROUPS ===")
        print(
            zero_p[
                [
                    "participant", "hand", "attempts_expected",
                    "mapping_unavailable", "metric_rows",
                    "qc_pass", "qc_warn", "sensitivity_rows",
                ]
            ].to_string(index=False)
        )

    zero_s = ph[ph["zero_sensitivity_rows"]]
    if len(zero_s):
        print("\n=== ZERO SENSITIVITY PARTICIPANT-HAND GROUPS ===")
        print(
            zero_s[
                [
                    "participant", "hand", "attempts_expected",
                    "mapping_unavailable", "metric_rows",
                    "qc_pass", "qc_warn",
                ]
            ].to_string(index=False)
        )

    print("\n=== OUTCOME AVAILABILITY ===")
    print(
        outcome[
            [
                "analysis_set", "outcome", "eligible_rows",
                "nonmissing_outcome_rows",
                "participants_with_nonmissing",
                "participant_hand_groups_with_nonmissing",
            ]
        ].to_string(index=False)
    )

    print()
    print("Wrote:", ph_path)
    print("Wrote:", outcome_path)
    print("Wrote:", balance_path)
    print("Wrote:", part_path)
    print()
    print(
        "Interpretation: zero-eligible strata are not data-processing failures if their "
        "source/mapping/QC status is documented; they determine which inferential models "
        "are estimable and which sensitivity analyses are required."
    )


if __name__ == "__main__":
    main()
