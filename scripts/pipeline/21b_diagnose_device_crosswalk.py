from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def norm_label(v: object) -> str:
    s = str(v).strip().upper()
    return re.sub(r"[^A-Z0-9]+", "", s)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Diagnose why mapping Device and workbook Device disagree. "
            "Reports the exact observed crosswalk and whether it is a strict bijection."
        )
    )
    ap.add_argument("--master", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    master = pd.read_csv(args.master, dtype=str).fillna("")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    required = {"attempt_id","hand","device","workbook_device"}
    missing = required - set(master.columns)
    if missing:
        raise RuntimeError("Master file is missing: " + ", ".join(sorted(missing)))

    # Collapse duplicate hand rows so the crosswalk is examined at attempt level.
    attempts = (
        master[["attempt_id","device","workbook_device"]]
        .drop_duplicates()
        .copy()
    )
    attempts["device_norm"] = attempts["device"].map(norm_label)
    attempts["workbook_device_norm"] = attempts["workbook_device"].map(norm_label)

    pairs = (
        attempts.groupby(
            ["device","workbook_device","device_norm","workbook_device_norm"],
            dropna=False
        )
        .size()
        .reset_index(name="attempts")
        .sort_values(["device","workbook_device"])
    )

    pair_path = outdir / "device_label_pair_counts.csv"
    pairs.to_csv(pair_path, index=False)

    crosstab = pd.crosstab(
        attempts["device"],
        attempts["workbook_device"],
        margins=True
    )
    crosstab_path = outdir / "device_label_crosstab.csv"
    crosstab.to_csv(crosstab_path)

    # A strict observed bijection means every mapping label corresponds to exactly
    # one workbook label and every workbook label corresponds to exactly one map label.
    left_counts = (
        attempts.groupby("device")["workbook_device"].nunique(dropna=False)
    )
    right_counts = (
        attempts.groupby("workbook_device")["device"].nunique(dropna=False)
    )
    strict_bijection = bool(
        len(left_counts)
        and len(right_counts)
        and (left_counts == 1).all()
        and (right_counts == 1).all()
    )

    proposed = pd.DataFrame()
    if strict_bijection:
        proposed = (
            attempts[["device","workbook_device"]]
            .drop_duplicates()
            .sort_values("device")
            .rename(columns={
                "device":"mapping_device",
                "workbook_device":"canonical_workbook_device",
            })
        )
        proposed["review_status"] = "PROPOSED_REVIEW_REQUIRED"
        proposed_path = outdir / "device_crosswalk_proposed.csv"
        proposed.to_csv(proposed_path, index=False)
    else:
        proposed_path = outdir / "device_crosswalk_proposed.csv"
        pd.DataFrame(columns=[
            "mapping_device","canonical_workbook_device","review_status"
        ]).to_csv(proposed_path, index=False)

    print("=== DEVICE LABEL CROSSWALK DIAGNOSTIC ===")
    print("Attempt-level rows checked:", len(attempts))
    print("Mapping device labels:", sorted(attempts["device"].unique()))
    print("Workbook device labels:", sorted(attempts["workbook_device"].unique()))
    print()
    print("Observed pair counts:")
    print(
        pairs[["device","workbook_device","attempts"]]
        .to_string(index=False)
    )
    print()
    print("Strict one-to-one observed crosswalk:", strict_bijection)
    print("Wrote:", pair_path)
    print("Wrote:", crosstab_path)
    print("Wrote:", proposed_path)

    if strict_bijection:
        print()
        print("PROPOSED CROSSWALK:")
        print(proposed[["mapping_device","canonical_workbook_device"]].to_string(index=False))
        print()
        print(
            "If these pairings are semantically correct, mark review_status=APPROVED "
            "in device_crosswalk_proposed.csv and use that file in the rebuild step."
        )
    else:
        print()
        print(
            "STOP: labels are not one-to-one. Do not auto-normalize; investigate the conflicting rows."
        )


if __name__ == "__main__":
    main()
