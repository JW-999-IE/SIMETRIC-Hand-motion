from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    "mapped_attempt_duration_sec",
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


def to_bool(series):
    return series.astype(str).str.lower().isin({"true","1","yes"})


def aggregate(master: pd.DataFrame, flag: str, label: str) -> pd.DataFrame:
    rows = []
    eligible = to_bool(master[flag])

    for (participant, hand), g in master.groupby(["participant","hand"]):
        use = g[eligible.loc[g.index]].copy()

        row = {
            "participant": participant,
            "hand": hand,
            "analysis_set": label,
            "attempts_expected": len(g),
            "attempts_mapping_unavailable": int(g["mapping_status"].eq("unavailable").sum()),
            "attempts_metric_row_present": int(g["_metric_merge"].eq("both").sum()),
            "attempts_qc_pass": int(g["qc_status"].eq("PASS").sum()),
            "attempts_qc_warn": int(g["qc_status"].eq("WARN_LOW_TRACK").sum()),
            "attempts_qc_fail": int(g["qc_status"].astype(str).str.startswith("FAIL").sum()),
            "attempts_analysis_eligible": len(use),
            "gopro_eligible": int(use["motion_source"].astype(str).str.startswith("gopro_").sum()),
            "cae_eligible": int(use["motion_source"].eq("cae_hand").sum()),
        }

        for metric in METRICS:
            if metric in use.columns:
                vals = pd.to_numeric(use[metric], errors="coerce")
                row[f"mean_{metric}"] = vals.mean()
                row[f"median_{metric}"] = vals.median()
            else:
                row[f"mean_{metric}"] = np.nan
                row[f"median_{metric}"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Build corrected participant×hand summaries with explicit expected, "
            "missing, QC and analysis-eligible counts. Means are computed only "
            "within the named analysis set."
        )
    )
    ap.add_argument("--master", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    master = pd.read_csv(args.master, dtype=str).fillna("")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    primary = aggregate(
        master,
        "analysis_set_primary_gopro",
        "primary_gopro_high_qc",
    )
    sensitivity = aggregate(
        master,
        "analysis_set_all_sources_sensitivity",
        "all_sources_sensitivity",
    )

    primary_path = outdir / "participant_hand_summary_primary.csv"
    sensitivity_path = outdir / "participant_hand_summary_sensitivity.csv"

    primary.to_csv(primary_path, index=False)
    sensitivity.to_csv(sensitivity_path, index=False)

    print("=== CORRECTED PARTICIPANT SUMMARIES COMPLETE ===")
    print("Primary participant-hand rows:", len(primary))
    print("Sensitivity participant-hand rows:", len(sensitivity))
    print("Wrote:", primary_path)
    print("Wrote:", sensitivity_path)
    print()
    print(
        "These replace the old Participant_Summary logic that counted FAIL/WARN "
        "rows as attempts 'with metrics'."
    )


if __name__ == "__main__":
    main()
