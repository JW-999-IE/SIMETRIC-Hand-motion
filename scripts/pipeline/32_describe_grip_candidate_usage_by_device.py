from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


GRIP_LABELS = [
    "EPG_CANDIDATE",
    "IPG_CANDIDATE",
    "TC_CANDIDATE",
    "NOT_GRIPPING_CANDIDATE",
    "UNCERTAIN",
]


def to_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin({"true", "1", "yes"})


def summarize(df: pd.DataFrame, group_cols: list[str], analysis_set: str):
    rows = []

    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]

    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key))

        total_time = g["segment_duration_sec"].sum()
        gripping = g[g["grip_candidate"].isin(
            ["EPG_CANDIDATE", "IPG_CANDIDATE", "TC_CANDIDATE"]
        )]
        gripping_time = gripping["segment_duration_sec"].sum()

        for label in GRIP_LABELS:
            x = g[g["grip_candidate"].eq(label)]
            dur = x["segment_duration_sec"].sum()
            rows.append({
                **base,
                "analysis_set": analysis_set,
                "grip_candidate": label,
                "segment_count": len(x),
                "duration_sec": dur,
                "share_of_all_segmented_time": (
                    dur / total_time if total_time > 0 else np.nan
                ),
                "share_of_classified_gripping_time": (
                    dur / gripping_time
                    if gripping_time > 0 and label in {
                        "EPG_CANDIDATE", "IPG_CANDIDATE", "TC_CANDIDATE"
                    }
                    else np.nan
                ),
                "mean_candidate_confidence": (
                    x["candidate_confidence"].mean() if len(x) else np.nan
                ),
                "attempts_represented": x["attempt_id"].nunique(),
                "participants_represented": x["participant"].nunique(),
            })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Descriptive grip-candidate usage by MST/ATG/CON. "
            "Time-weighted summaries only; no inferential claim because grip labels "
            "remain screening-only until manual validation."
        )
    )
    ap.add_argument("--grip-segments", required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    grip = pd.read_csv(
        args.grip_segments,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    master = pd.read_csv(
        args.master,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )

    required_grip = {
        "attempt_id", "participant", "hand", "motion_source",
        "segment_duration_sec", "grip_candidate", "candidate_confidence",
    }
    missing = required_grip - set(grip.columns)
    if missing:
        raise RuntimeError("Grip file missing columns: " + ", ".join(sorted(missing)))

    required_master = {
        "attempt_id", "hand", "device",
        "analysis_main",
        "analysis_highqc_sensitivity",
        "analysis_recovery_sensitivity",
    }
    missing = required_master - set(master.columns)
    if missing:
        raise RuntimeError("Master missing columns: " + ", ".join(sorted(missing)))

    grip["segment_duration_sec"] = pd.to_numeric(
        grip["segment_duration_sec"], errors="coerce"
    )
    grip["candidate_confidence"] = pd.to_numeric(
        grip["candidate_confidence"], errors="coerce"
    )
    grip = grip.dropna(subset=["segment_duration_sec"]).copy()

    map_cols = [
        "attempt_id", "hand", "device",
        "analysis_main",
        "analysis_highqc_sensitivity",
        "analysis_recovery_sensitivity",
    ]
    mm = master[map_cols].drop_duplicates(["attempt_id", "hand"])

    merged = grip.merge(
        mm,
        on=["attempt_id", "hand"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched = merged["_merge"].ne("both")
    if unmatched.any():
        print("STOP: grip segments failed to join to master:")
        print(
            merged.loc[
                unmatched,
                ["attempt_id", "participant", "hand", "motion_source"]
            ].drop_duplicates().head(50).to_string(index=False)
        )
        raise SystemExit(2)

    merged = merged.drop(columns="_merge")

    # All v3 mapped grip candidates.
    all_summary = summarize(
        merged,
        ["device"],
        "all_available_grip_segments",
    )

    # Main analysis population, kept descriptive only.
    main = merged[to_bool(merged["analysis_main"])].copy()
    main_summary = summarize(
        main,
        ["device"],
        "main_gopro_ge60_descriptive",
    )

    # Hand-specific main population.
    main_hand = summarize(
        main,
        ["device", "hand"],
        "main_gopro_ge60_descriptive",
    )

    # High-QC descriptive sensitivity.
    highqc = merged[to_bool(merged["analysis_highqc_sensitivity"])].copy()
    highqc_summary = summarize(
        highqc,
        ["device"],
        "highqc_gopro_ge80_descriptive",
    )

    # Recovery descriptive sensitivity.
    recovery = merged[to_bool(merged["analysis_recovery_sensitivity"])].copy()
    recovery_summary = summarize(
        recovery,
        ["device"],
        "recovery_allsource_ge60_descriptive",
    )

    summary = pd.concat(
        [all_summary, main_summary, highqc_summary, recovery_summary],
        ignore_index=True,
    )

    summary_path = outdir / "grip_candidate_usage_by_device.csv"
    hand_path = outdir / "grip_candidate_usage_by_device_hand_main.csv"
    merged_path = outdir / "grip_segments_with_device_and_analysis_flags.csv"

    summary.to_csv(summary_path, index=False)
    main_hand.to_csv(hand_path, index=False)
    merged.to_csv(merged_path, index=False)

    # Winner by time for each analysis set/device.
    winners = (
        summary[
            summary["grip_candidate"].isin(
                ["EPG_CANDIDATE", "IPG_CANDIDATE", "TC_CANDIDATE"]
            )
        ]
        .sort_values(
            ["analysis_set", "device", "duration_sec"],
            ascending=[True, True, False],
        )
        .groupby(["analysis_set", "device"], as_index=False)
        .first()
    )

    winner_path = outdir / "grip_candidate_most_used_by_device.csv"
    winners.to_csv(winner_path, index=False)

    print("=== GRIP-CANDIDATE DESCRIPTIVE SUMMARY ===")
    print(
        "IMPORTANT: these are automated screening candidates, not validated grip labels."
    )
    print()

    for aset in [
        "main_gopro_ge60_descriptive",
        "highqc_gopro_ge80_descriptive",
        "recovery_allsource_ge60_descriptive",
        "all_available_grip_segments",
    ]:
        x = winners[winners["analysis_set"].eq(aset)]
        if x.empty:
            continue
        print(f"=== {aset} ===")
        print(
            x[
                [
                    "device", "grip_candidate",
                    "duration_sec",
                    "share_of_all_segmented_time",
                    "share_of_classified_gripping_time",
                    "participants_represented",
                ]
            ].to_string(index=False)
        )
        print()

    print("Wrote:", summary_path)
    print("Wrote:", hand_path)
    print("Wrote:", winner_path)
    print("Wrote:", merged_path)
    print()
    print(
        "Interpretation rule: use duration-weighted grip-candidate shares descriptively. "
        "Do not report IPG/EPG/TC as validated behavioural outcomes until the manual "
        "validation sample has been reviewed and classifier performance quantified."
    )


if __name__ == "__main__":
    main()
