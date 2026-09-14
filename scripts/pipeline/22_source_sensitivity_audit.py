from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CORE_METRICS = [
    "mean_accel_norm_s2",
    "rms_accel_norm_s2",
    "peak_accel_norm_s2",
    "sparc",
    "log_dimensionless_jerk",
    "velocity_peak_count",
    "path_length_norm",
    "path_efficiency",
    "stillness_fraction",
    "tremor_rms_norm",
    "tremor_dominant_hz",
    "first_sustained_hand_movement_delay_sec",
]


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Create source-stratified descriptive/sensitivity summaries. "
            "This does not claim GoPro and CAE are geometrically interchangeable."
        )
    )
    ap.add_argument("--master", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.master)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for metric in CORE_METRICS:
        if metric not in df.columns:
            continue
        x = pd.to_numeric(df[metric], errors="coerce")
        tmp = df.assign(_value=x)
        for (source, hand), g in tmp.groupby(["motion_source","hand"], dropna=False):
            vals = g["_value"].dropna()
            rows.append({
                "metric": metric,
                "motion_source": source,
                "hand": hand,
                "rows_total": len(g),
                "rows_nonmissing_metric": len(vals),
                "participants": g["participant"].nunique(),
                "mean": vals.mean() if len(vals) else np.nan,
                "sd": vals.std(ddof=1) if len(vals)>1 else np.nan,
                "median": vals.median() if len(vals) else np.nan,
                "q25": vals.quantile(.25) if len(vals) else np.nan,
                "q75": vals.quantile(.75) if len(vals) else np.nan,
                "min": vals.min() if len(vals) else np.nan,
                "max": vals.max() if len(vals) else np.nan,
            })

    summary = pd.DataFrame(rows)
    summary_path = outdir / "source_stratified_metric_summary.csv"
    summary.to_csv(summary_path, index=False)

    counts = (
        df.groupby(
            ["motion_source","hand","analysis_set_primary_gopro",
             "analysis_set_all_sources_sensitivity"],
            dropna=False
        )
        .size()
        .reset_index(name="rows")
    )
    counts_path = outdir / "source_analysis_set_counts.csv"
    counts.to_csv(counts_path, index=False)

    participant_source = (
        df.groupby(["participant","hand","motion_source"], dropna=False)
        .size().reset_index(name="rows")
    )
    ps_path = outdir / "participant_hand_source_counts.csv"
    participant_source.to_csv(ps_path, index=False)

    note = outdir / "source_sensitivity_README.txt"
    note.write_text(
        "Interpretation rul<SET_YOUR_ANALYSIS_ROOT>"
        "1. Primary inferential motion analyses use rows where analysis_set_primary_gopro=True.\n"
        "2. CAE-HAND is secondary recovery and should not be assumed geometrically equivalent to GoPro.\n"
        "3. analysis_set_all_sources_sensitivity=True may be used only as a sensitivity analysis with motion_source retained/adjusted.\n"
        "4. Absolute normalized X/Y coordinates are viewpoint dependent and should not be directly pooled across camera systems.\n",
        encoding="utf-8",
    )

    print("=== SOURCE SENSITIVITY AUDIT COMPLETE ===")
    print("Wrote:", summary_path)
    print("Wrote:", counts_path)
    print("Wrote:", ps_path)
    print("Wrote:", note)
    print()
    print("Motion-source counts:")
    print(df["motion_source"].fillna("<missing>").value_counts().to_string())
    print()
    print("Primary GoPro rows:", int(df["analysis_set_primary_gopro"].astype(bool).sum()))
    print("All-source sensitivity rows:", int(df["analysis_set_all_sources_sensitivity"].astype(bool).sum()))


if __name__ == "__main__":
    main()
