from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def bool_col(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin({"true", "1", "yes"})


def workbook_family(label: object) -> str:
    s = str(label).strip().upper()
    m = re.match(r"^(DEVICE\d+)-\d+$", s)
    if not m:
        return ""
    return m.group(1)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Rebuild the 828-row statistical master using the observed workbook "
            "device-family hierarchy. Workbook detail labels such as DEVICE1-1..5 "
            "are reduced to DEVICE1/DEVICE2/DEVICE3 for consistency checking against "
            "the mapping's categorical device labels, while preserving the detailed "
            "workbook labels for audit."
        )
    )
    ap.add_argument("--master", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    master_path = Path(args.master)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(master_path, dtype=str).fillna("")

    required = {
        "attempt_id", "participant", "hand",
        "device", "workbook_device",
        "round_match", "_workbook_merge",
        "mapping_status", "_metric_merge",
        "analysis_set_primary_gopro",
        "analysis_set_all_sources_sensitivity",
        "analysis_set_cae_recovery_sensitivity",
        "analysis_set_grip_inference",
    }
    missing = required - set(master.columns)
    if missing:
        raise RuntimeError(
            "Master file is missing required columns: "
            + ", ".join(sorted(missing))
        )

    master["device_from_mapping_original"] = master["device"]
    master["workbook_device_detail"] = master["workbook_device"]
    master["workbook_device_family"] = master["workbook_device"].map(workbook_family)

    malformed = master[master["workbook_device_family"].eq("")]
    if len(malformed):
        print("STOP: some workbook device labels do not match DEVICE<number>-<number>.")
        print(
            malformed[["attempt_id", "workbook_device"]]
            .drop_duplicates()
            .head(50)
            .to_string(index=False)
        )
        raise SystemExit(2)

    # Work at attempt level so left/right duplicate rows do not distort counts.
    attempts = master[
        ["attempt_id", "device_from_mapping_original", "workbook_device_family"]
    ].drop_duplicates()

    # Require a strict one-to-one relationship between workbook family and
    # mapping category. This is inferred from the actual joined data, not hard-coded.
    fam_to_map = (
        attempts.groupby("workbook_device_family")["device_from_mapping_original"]
        .nunique()
    )
    map_to_fam = (
        attempts.groupby("device_from_mapping_original")["workbook_device_family"]
        .nunique()
    )

    strict_bijection = bool(
        len(fam_to_map)
        and len(map_to_fam)
        and (fam_to_map == 1).all()
        and (map_to_fam == 1).all()
    )

    crosswalk = (
        attempts[
            ["workbook_device_family", "device_from_mapping_original"]
        ]
        .drop_duplicates()
        .sort_values("workbook_device_family")
        .rename(columns={
            "workbook_device_family": "workbook_device_family",
            "device_from_mapping_original": "canonical_device_category",
        })
    )

    crosswalk_path = outdir / "device_family_crosswalk_inferred.csv"
    crosswalk.to_csv(crosswalk_path, index=False)

    print("=== DEVICE FAMILY HIERARCHY CHECK ===")
    print("Attempt-level rows checked:", len(attempts))
    print()
    print("Observed family crosswalk:")
    print(crosswalk.to_string(index=False))
    print()
    print("Strict family-to-category bijection:", strict_bijection)

    if not strict_bijection:
        print("STOP: workbook device families do not map one-to-one to mapping categories.")
        raise SystemExit(2)

    mapping = dict(
        zip(
            crosswalk["workbook_device_family"],
            crosswalk["canonical_device_category"],
        )
    )

    master["device_category_from_workbook_family"] = (
        master["workbook_device_family"].map(mapping)
    )
    master["device_match"] = (
        master["device_category_from_workbook_family"]
        == master["device_from_mapping_original"]
    )

    mismatches = master[~master["device_match"]]
    if len(mismatches):
        print("STOP: family-level device mismatches remain.")
        print(
            mismatches[
                [
                    "attempt_id",
                    "device_from_mapping_original",
                    "workbook_device_detail",
                    "workbook_device_family",
                    "device_category_from_workbook_family",
                ]
            ]
            .drop_duplicates()
            .head(50)
            .to_string(index=False)
        )
        raise SystemExit(2)

    # Canonical modelling variable remains MST/ATG/CON.
    master["device"] = master["device_category_from_workbook_family"]

    # Structural QC.
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

    out = outdir / "statistical_master_attempt_hand_828_v2.csv"
    qc_path = outdir / "statistical_master_qc_summary_v2.csv"

    master.to_csv(out, index=False)
    qc.to_csv(qc_path, index=False)

    blockers = (
        workbook_unmatched
        + int((~round_ok).sum())
        + int((~device_ok).sum())
    )

    print()
    print("=== STATISTICAL MASTER V2 COMPLETE ===")
    print("Rows:", len(master))
    print("Participants:", master["participant"].nunique())
    print("Workbook unmatched:", workbook_unmatched)
    print("Round mismatches:", int((~round_ok).sum()))
    print("Device mismatches:", int((~device_ok).sum()))
    print("Structural blockers:", blockers)
    print("Primary GoPro analysis rows:",
          int(bool_col(master["analysis_set_primary_gopro"]).sum()))
    print("All-source sensitivity rows:",
          int(bool_col(master["analysis_set_all_sources_sensitivity"]).sum()))
    print()
    print("Wrote:", out)
    print("Wrote:", qc_path)
    print("Wrote:", crosswalk_path)

    if blockers:
        print("STOP before modelling.")
    else:
        print(
            "PASS: device hierarchy is internally consistent and the 828-row "
            "master has zero structural blockers."
        )


if __name__ == "__main__":
    main()
