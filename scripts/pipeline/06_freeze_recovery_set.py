from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from common import ensure_recovery_root, load_config, parse_bool, read_table, sha256_file, write_csv


def validated_sub60_flags(config: dict, eligibility: pd.DataFrame, summary: pd.DataFrame) -> pd.Series:
    freeze = config.get("freeze") or {}
    if not bool(freeze.get("allow_validated_sub60", False)) or summary.empty:
        return pd.Series(False, index=eligibility.index)
    max_gap = float(freeze.get("validated_sub60_max_gap_fraction", 0.20))
    min_bins = float(freeze.get("validated_sub60_min_temporal_bin_coverage", 0.70))
    results: list[bool] = []
    for _, row in eligibility.iterrows():
        if not bool(row.get("sub60_validation_candidate", False)):
            results.append(False)
            continue
        candidates = summary[
            summary["source"].eq(row["source"])
            & summary["target_hand"].eq(row["target_hand"])
            & summary["requested_coverage"].le(float(row["target_detection_coverage"]))
        ]
        if candidates.empty:
            results.append(False)
            continue
        level = candidates["requested_coverage"].max()
        level_rows = candidates[candidates["requested_coverage"].eq(level)]
        validation_pass = level_rows["validation_pass"].map(parse_bool).all() and {
            "distributed",
            "single_gap",
        }.issubset(set(level_rows["pattern"]))
        temporal_pass = (
            float(row.get("longest_gap_fraction", np.inf)) <= max_gap
            and float(row.get("temporal_bin_coverage", -np.inf)) >= min_bins
        )
        results.append(bool(validation_pass and temporal_pass))
    return pd.Series(results, index=eligibility.index)


def choose_frozen_sources(eligibility: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    keys = ["attempt_id", "participant", "configuration", "target_hand"]
    priority = [
        ("primary_trajectory_eligible", "primary_gopro60"),
        ("validated_sub60", "recovery_gopro_validated_sub60"),
        ("cae_trajectory_recovery", "recovery_cae60"),
        ("validated_cae_sub60", "recovery_cae_validated_sub60"),
    ]
    for key, group in eligibility.groupby(keys, dropna=False):
        chosen = None
        label = "ineligible"
        for flag, analysis_set in priority:
            candidates = group[group[flag].astype(bool)]
            if len(candidates) > 1:
                raise ValueError(f"Multiple {analysis_set} candidates for {key}; mapping is not frozen")
            if len(candidates) == 1:
                chosen = candidates.iloc[0]
                label = analysis_set
                break
        rows.append(
            {
                "attempt_id": key[0],
                "participant": key[1],
                "configuration": key[2],
                "target_hand": key[3],
                "analysis_set": label,
                "motion_source": "" if chosen is None else chosen["source"],
                "video_path": "" if chosen is None else chosen.get("video_path", ""),
                "target_detection_coverage": np.nan if chosen is None else chosen.get("target_detection_coverage", np.nan),
                "temporal_bin_coverage": np.nan if chosen is None else chosen.get("temporal_bin_coverage", np.nan),
                "longest_gap_fraction": np.nan if chosen is None else chosen.get("longest_gap_fraction", np.nan),
                "landmark_path": "" if chosen is None else chosen.get("landmark_path", ""),
                "use_in_primary_gopro_analysis": label == "primary_gopro60",
                "use_in_expanded_recovery_analysis": label != "ineligible",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze an outcome-blind, source-labelled recovery mapping table."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--eligibility", default="")
    parser.add_argument("--sparse-summary", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    if not bool((config.get("freeze") or {}).get("rules_frozen", False)):
        raise ValueError(
            "Set freeze.rules_frozen: true only after the mapping and QC rules are approved, before outcome modelling."
        )
    recovery_root = ensure_recovery_root(config)
    eligibility_path = (
        Path(args.eligibility).expanduser().resolve()
        if args.eligibility
        else recovery_root / "03_eligibility_audit" / "attempt_hand_source_eligibility.csv"
    )
    sparse_path = (
        Path(args.sparse_summary).expanduser().resolve()
        if args.sparse_summary
        else recovery_root / "04_sparse_validation" / "sparse_validation_summary.csv"
    )
    eligibility = read_table(eligibility_path)
    summary = read_table(sparse_path) if sparse_path.exists() else pd.DataFrame()

    validated = validated_sub60_flags(config, eligibility, summary)
    eligibility["validated_sub60"] = (
        validated
        & eligibility["source"].eq("gopro_" + eligibility["target_hand"].astype(str))
    )
    eligibility["validated_cae_sub60"] = validated & eligibility["source"].eq("cae_hand")
    frozen = choose_frozen_sources(eligibility)

    output_dir = recovery_root / "05_frozen_recovery_set"
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = write_csv(frozen, output_dir / "frozen_attempt_hand_source_map.csv")
    write_csv(
        eligibility,
        output_dir / "frozen_source_eligibility_with_validated_sub60.csv",
    )
    counts = (
        frozen.groupby(["analysis_set", "target_hand", "configuration"], dropna=False)
        .agg(attempts=("attempt_id", "nunique"), participants=("participant", "nunique"))
        .reset_index()
    )
    write_csv(counts, output_dir / "frozen_recovery_counts.csv")

    rules_snapshot = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "eligibility": config.get("eligibility") or {},
        "trajectory_qc": config.get("trajectory_qc") or {},
        "sparse_validation": config.get("sparse_validation") or {},
        "freeze": config.get("freeze") or {},
        "scientific_constraints": {
            "gopro_primary": True,
            "cae_hand_recovery_only": True,
            "cae_room_excluded": True,
            "left_right_separate": True,
            "motion_source_retained": True,
            "opposite_camera_attempts_not_inferred": True,
            "workbook_timestamps_reference_only": True,
        },
    }
    rules_path = output_dir / "frozen_rules.yaml"
    with rules_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(rules_snapshot, handle, sort_keys=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            str(eligibility_path): sha256_file(eligibility_path),
            **({str(sparse_path): sha256_file(sparse_path)} if sparse_path.exists() else {}),
        },
        "outputs": {
            str(frozen_path): sha256_file(frozen_path),
            str(rules_path): sha256_file(rules_path),
        },
    }
    manifest_path = output_dir / "freeze_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Frozen source map and rules written to {output_dir}")
    print("Do not change these rules after inspecting configuration effects or P values.")


if __name__ == "__main__":
    main()
