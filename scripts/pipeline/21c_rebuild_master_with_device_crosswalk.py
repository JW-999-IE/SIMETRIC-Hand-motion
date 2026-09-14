from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def bool_col(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin({"true","1","yes"})


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Rebuild device consistency in an existing 828-row statistical master "
            "using an explicitly reviewed device-label crosswalk. The workbook device "
            "is retained as canonical; mapping device is preserved for audit."
        )
    )
    ap.add_argument("--master", required=True)
    ap.add_argument("--crosswalk", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    master_path = Path(args.master)
    crosswalk_path = Path(args.crosswalk)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(master_path, dtype=str).fillna("")
    cross = pd.read_csv(crosswalk_path, dtype=str).fillna("")

    required = {
        "mapping_device",
        "canonical_workbook_device",
        "review_status",
    }
    missing = required - set(cross.columns)
    if missing:
        raise RuntimeError(
            "Crosswalk is missing: " + ", ".join(sorted(missing))
        )

    if cross.empty:
        raise RuntimeError("Crosswalk is empty.")

    bad_status = cross[
        cross["review_status"].astype(str).str.upper().ne("APPROVED")
    ]
    if len(bad_status):
        print("STOP: every crosswalk row must be explicitly APPROVED.")
        print(bad_status.to_string(index=False))
        raise SystemExit(2)

    if cross["mapping_device"].duplicated().any():
        raise RuntimeError("Crosswalk contains duplicate mapping_device labels.")
    if cross["canonical_workbook_device"].duplicated().any():
        raise RuntimeError("Crosswalk is not one-to-one on canonical_workbook_device.")

    mapping_dict = dict(
        zip(cross["mapping_device"], cross["canonical_workbook_device"])
    )

    master["device_from_mapping_original"] = master["device"]
    master["device_canonical"] = master["device"].map(mapping_dict).fillna("")

    # Workbook Device is authoritative for the joined attempt record.
    master["device_match"] = (
        master["device_canonical"].astype(str).str.strip().str.upper()
        == master["workbook_device"].astype(str).str.strip().str.upper()
    )

    unknown = master[
        master["device_canonical"].eq("")
        & master["device"].ne("")
    ]
    mismatches = master[~master["device_match"]]

    if len(unknown):
        print("STOP: crosswalk did not cover all mapping device labels.")
        print(
            unknown[["attempt_id","device","workbook_device"]]
            .drop_duplicates()
            .to_string(index=False)
        )
        raise SystemExit(2)

    if len(mismatches):
        print("STOP: approved crosswalk still leaves device mismatches.")
        print(
            mismatches[
                ["attempt_id","device","device_canonical","workbook_device"]
            ]
            .drop_duplicates()
            .head(50)
            .to_string(index=False)
        )
        raise SystemExit(2)

    # Preserve the original mapping label but expose the workbook coding as
    # canonical for modelling.
    master["device"] = master["workbook_device"]

    out = outdir / "statistical_master_attempt_hand_828_v2.csv"
    master.to_csv(out, index=False)

    # Recalculate the structural QC summary.
    workbook_unmatched = int((master["_workbook_merge"] != "both").sum())
    round_ok = bool_col(master["round_match"])
    device_ok = bool_col(master["device_match"])

    qc = pd.DataFrame({
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
            workbook_unmatched,
            int((~round_ok).sum()),
            int((~device_ok).sum()),
            int(master["mapping_status"].eq("unavailable").sum()),
            int(master["_metric_merge"].eq("both").sum()),
            int(bool_col(master["analysis_set_primary_gopro"]).sum()),
            int(bool_col(master["analysis_set_all_sources_sensitivity"]).sum()),
            int(bool_col(master["analysis_set_cae_recovery_sensitivity"]).sum()),
            int(bool_col(master["analysis_set_grip_inference"]).sum()),
        ],
    })

    qc_path = outdir / "statistical_master_qc_summary_v2.csv"
    qc.to_csv(qc_path, index=False)

    blockers = (
        workbook_unmatched
        + int((~round_ok).sum())
        + int((~device_ok).sum())
    )

    print("=== STATISTICAL MASTER DEVICE REBUILD COMPLETE ===")
    print("Rows:", len(master))
    print("Participants:", master["participant"].nunique())
    print("Workbook unmatched:", workbook_unmatched)
    print("Round mismatches:", int((~round_ok).sum()))
    print("Device mismatches:", int((~device_ok).sum()))
    print("Structural blockers:", blockers)
    print()
    print("Wrote:", out)
    print("Wrote:", qc_path)

    if blockers:
        print("STOP before modelling.")
    else:
        print("PASS: 828-row master is structurally consistent and ready for downstream QC/statistical preparation.")


if __name__ == "__main__":
    main()
