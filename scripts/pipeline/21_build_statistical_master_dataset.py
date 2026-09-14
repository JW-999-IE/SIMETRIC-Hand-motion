from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


EXCLUDED = {"P7","P8","P14"}


def p_simple(v):
    m = re.search(r"(\d+)", str(v))
    return f"P{int(m.group(1))}" if m else str(v)


def p_norm(v):
    m = re.search(r"(\d+)", str(v))
    return f"P{int(m.group(1)):02d}" if m else str(v)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str).fillna("")


def bool_col(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin({"true","1","yes"})


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Build the complete 828-row SIMETRIC statistical master dataset by "
            "left-joining corrected metrics onto every expected attempt×hand row, "
            "with explicit missingness, FTIS, source, QC and analysis-set flags."
        )
    )
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--metrics-v3", required=True)
    ap.add_argument("--insertion-workbook", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--participant-covariates",
        help=(
            "Optional CSV keyed by participant, e.g. expertise/device-experience variables. "
            "Columns are merged without inventing missing values."
        ),
    )
    args = ap.parse_args()

    mapping = read_csv(Path(args.mapping))
    metrics = read_csv(Path(args.metrics_v3))
    insert = pd.read_excel(args.insertion_workbook, sheet_name=0, dtype=object)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    mapping["participant_key"] = mapping["participant"].map(p_simple)
    metrics["participant_key"] = metrics["participant"].map(p_simple)
    mapping["hand"] = mapping["hand"].astype(str).str.lower()
    metrics["hand"] = metrics["hand"].astype(str).str.lower()

    mapping = mapping[~mapping["participant_key"].isin(EXCLUDED)].copy()
    metrics = metrics[~metrics["participant_key"].isin(EXCLUDED)].copy()

    # Authoritative workbook attempt sequence is the row order within Id.
    insert = insert.copy()
    insert["participant_key"] = insert["Id"].map(p_simple)
    insert["attempt_sequence"] = (
        insert.groupby("participant_key", sort=False).cumcount() + 1
    )
    insert["attempt_id"] = (
        insert["participant_key"]
        + "_T"
        + insert["attempt_sequence"].astype(int).astype(str).str.zfill(2)
    )

    insert_small = insert[
        [
            "attempt_id","participant_key","attempt_sequence",
            "Round","Device","FTIS","Start time (ms)","End Time (ms)",
            "Duration (ms)","Needle In (ms)","Wire In (ms)",
            "Needle Out (ms)","Catheter Over Needle (ms)","Wire Out (ms)"
        ]
    ].copy()

    insert_small = insert_small.rename(columns={
        "Round":"workbook_round",
        "Device":"workbook_device",
        "FTIS":"ftis",
        "Start time (ms)":"workbook_start_ms",
        "End Time (ms)":"workbook_end_ms",
        "Duration (ms)":"workbook_duration_ms",
        "Needle In (ms)":"needle_in_ms",
        "Wire In (ms)":"wire_in_ms",
        "Needle Out (ms)":"needle_out_ms",
        "Catheter Over Needle (ms)":"catheter_over_needle_ms",
        "Wire Out (ms)":"wire_out_ms",
    })
    insert_small["success_binary"] = (
        insert_small["ftis"].astype(str).str.upper().map({"Y":1,"N":0})
    )

    # Full denominator first.
    master = mapping.merge(
        insert_small.drop(columns=["participant_key","attempt_sequence"]),
        on="attempt_id",
        how="left",
        validate="many_to_one",
        indicator="_workbook_merge",
    )

    # Metric rows should be unique attempt×hand.
    dup = metrics.duplicated(["attempt_id","hand"], keep=False)
    if dup.any():
        print("STOP: duplicate corrected metric rows detected:")
        print(
            metrics.loc[dup,["attempt_id","participant","hand"]]
            .sort_values(["attempt_id","hand"])
            .to_string(index=False)
        )
        raise SystemExit(2)

    metric_drop = [
        c for c in ["participant","participant_key","attempt_sequence","round","device"]
        if c in metrics.columns
    ]
    master = master.merge(
        metrics.drop(columns=metric_drop),
        on=["attempt_id","hand"],
        how="left",
        validate="one_to_one",
        indicator="_metric_merge",
        suffixes=("","_metric"),
    )

    # Explicit missingness reason.
    def reason(row):
        if str(row.get("mapping_status","")) == "unavailable":
            return "mapping_unavailable"
        if row["_metric_merge"] == "left_only":
            return "no_metric_row"
        qc = str(row.get("qc_status",""))
        if qc.startswith("FAIL"):
            return qc
        if qc == "WARN_LOW_TRACK":
            return "tracking_warn"
        return ""

    master["missingness_reason"] = master.apply(reason, axis=1)

    primary_metric = (
        bool_col(master.get("analysis_eligible_primary", pd.Series(False, index=master.index)))
    )
    sensitivity_metric = (
        bool_col(master.get("analysis_eligible_sensitivity", pd.Series(False, index=master.index)))
    )
    source = master.get("motion_source", pd.Series("", index=master.index)).astype(str)

    # Primary inference: high-QC GoPro only. CAE remains recovery/sensitivity.
    master["analysis_set_primary_gopro"] = (
        primary_metric & source.str.startswith("gopro_")
    )
    master["analysis_set_all_sources_sensitivity"] = sensitivity_metric
    master["analysis_set_cae_recovery_sensitivity"] = (
        sensitivity_metric & source.eq("cae_hand")
    )
    master["analysis_set_tremor_exploratory"] = (
        primary_metric & source.str.startswith("gopro_")
    )
    master["analysis_set_grip_inference"] = False
    master["grip_analysis_status"] = (
        "screening_only_manual_validation_required"
    )

    # Workbook-vs-map consistency checks.
    master["round_match"] = (
        master["round"].astype(str).str.strip()
        == master["workbook_round"].astype(str).str.strip()
    )
    master["device_match"] = (
        master["device"].astype(str).str.strip().str.upper()
        == master["workbook_device"].astype(str).str.strip().str.upper()
    )

    # Optional participant-level covariates.
    if args.participant_covariates:
        cov = read_csv(Path(args.participant_covariates))
        if "participant" not in cov.columns:
            raise RuntimeError("participant-covariates CSV must contain a 'participant' column.")
        cov["participant_key"] = cov["participant"].map(p_simple)
        cov = cov.drop(columns=["participant"]).drop_duplicates("participant_key")
        master = master.merge(
            cov, on="participant_key", how="left", validate="many_to_one"
        )

    master["participant"] = master["participant_key"].map(p_norm)

    # Stable order.
    master["_p"] = master["participant_key"].str.extract(r"(\d+)")[0].astype(int)
    master["_seq"] = pd.to_numeric(master["attempt_sequence"], errors="coerce")
    hand_order = master["hand"].map({"left":0,"right":1}).fillna(9)
    master["_h"] = hand_order
    master = master.sort_values(["_p","_seq","_h"]).drop(columns=["_p","_seq","_h"])

    out = outdir / "statistical_master_attempt_hand_828.csv"
    master.to_csv(out, index=False)

    qc_summary = pd.DataFrame({
        "item": [
            "rows_total",
            "participants",
            "workbook_unmatched_rows",
            "round_mismatches",
            "device_mismatches",
            "mapping_unavailable",
            "metric_rows_present",
            "primary_gopro_rows",
            "all_source_sensitivity_rows",
            "cae_recovery_sensitivity_rows",
            "grip_inference_rows",
        ],
        "count": [
            len(master),
            master["participant"].nunique(),
            int((master["_workbook_merge"]!="both").sum()),
            int((~master["round_match"]).sum()),
            int((~master["device_match"]).sum()),
            int(master["mapping_status"].eq("unavailable").sum()),
            int(master["_metric_merge"].eq("both").sum()),
            int(master["analysis_set_primary_gopro"].sum()),
            int(master["analysis_set_all_sources_sensitivity"].sum()),
            int(master["analysis_set_cae_recovery_sensitivity"].sum()),
            int(master["analysis_set_grip_inference"].sum()),
        ],
    })
    qc_path = outdir / "statistical_master_qc_summary.csv"
    qc_summary.to_csv(qc_path, index=False)

    print("=== STATISTICAL MASTER DATASET COMPLETE ===")
    print("Rows:", len(master))
    print("Participants:", master["participant"].nunique())
    print("Workbook unmatched:", int((master["_workbook_merge"]!="both").sum()))
    print("Round mismatches:", int((~master["round_match"]).sum()))
    print("Device mismatches:", int((~master["device_match"]).sum()))
    print("Mapping unavailable:", int(master["mapping_status"].eq("unavailable").sum()))
    print("Primary GoPro analysis rows:", int(master["analysis_set_primary_gopro"].sum()))
    print("All-source sensitivity rows:", int(master["analysis_set_all_sources_sensitivity"].sum()))
    print("CAE sensitivity rows:", int(master["analysis_set_cae_recovery_sensitivity"].sum()))
    print()
    print("Wrote:", out)
    print("Wrote:", qc_path)

    blockers = (
        int((master["_workbook_merge"]!="both").sum())
        + int((~master["round_match"]).sum())
        + int((~master["device_match"]).sum())
    )
    print("Master-data structural blockers:", blockers)
    if blockers:
        print("STOP before modelling: inspect workbook merge/round/device mismatches.")
    else:
        print("PASS: complete 828-row denominator with explicit missingness and analysis-set flags.")


if __name__ == "__main__":
    main()
