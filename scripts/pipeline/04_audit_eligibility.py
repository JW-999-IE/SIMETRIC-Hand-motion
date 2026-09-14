from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    MAPPING_ALIASES,
    QC_ALIASES,
    assert_unique,
    canonicalize_columns,
    ensure_recovery_root,
    load_config,
    normalize_participant,
    normalize_source,
    read_table,
    resolve_path,
    safe_float,
    source_target_hand,
    write_csv,
)


RESOLVED_STATUSES = {"resolved", "mapped", "confirmed", "complete"}
VISIBLE_STATES = {"complete", "partial", "visible", ""}


def coverage_fraction(value: object) -> float:
    number = safe_float(value)
    if np.isnan(number):
        return number
    return number / 100.0 if number > 1.0 else number


def hard_qc_failed(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"pass", "passed", "ok", "none", "false", "0", "no", "n"}:
        return False
    if text in {"fail", "failed", "hard", "true", "1", "yes", "y", "review_fail"}:
        return True
    return False


def pass_value(value: object) -> bool | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    text = str(value).strip().lower()
    if text in {"pass", "passed", "ok", "true", "1", "yes", "y"}:
        return True
    if text in {"fail", "failed", "false", "0", "no", "n"}:
        return False
    return None


def load_mapping(config: dict, path: Path) -> pd.DataFrame:
    mapping = canonicalize_columns(
        read_table(path),
        (config.get("column_maps") or {}).get("mapping"),
        MAPPING_ALIASES,
        required=["attempt_id", "participant", "configuration", "source", "target_hand", "mapping_status"],
    )
    for column in MAPPING_ALIASES:
        if column not in mapping:
            mapping[column] = ""
    mapping["attempt_id"] = mapping["attempt_id"].astype(str).str.strip()
    mapping["participant"] = mapping["participant"].map(normalize_participant)
    mapping["source"] = mapping["source"].map(normalize_source)
    mapping["target_hand"] = mapping["target_hand"].astype(str).str.strip().str.lower()
    mapping["mapping_status"] = mapping["mapping_status"].fillna("").astype(str).str.strip().str.lower()
    mapping["coverage_state"] = mapping["coverage_state"].fillna("").astype(str).str.strip().str.lower()
    return mapping


def load_qc(config: dict, path: Path) -> pd.DataFrame:
    qc = canonicalize_columns(
        read_table(path),
        (config.get("column_maps") or {}).get("qc"),
        QC_ALIASES,
        required=["attempt_id", "source", "target_hand", "target_detection_coverage"],
    )
    for column in QC_ALIASES:
        if column not in qc:
            qc[column] = np.nan
    qc["attempt_id"] = qc["attempt_id"].astype(str).str.strip()
    qc["source"] = qc["source"].map(normalize_source)
    qc["target_hand"] = qc["target_hand"].astype(str).str.strip().str.lower()
    for column in [
        "target_detection_coverage",
        "target_motion_coverage",
        "temporal_bin_coverage",
        "longest_gap_fraction",
        "identity_consistency",
        "ambiguous_fraction",
    ]:
        qc[column] = qc[column].map(coverage_fraction)
    qc["finite_wrist_points"] = pd.to_numeric(qc["finite_wrist_points"], errors="coerce")
    qc["side_switch_count"] = pd.to_numeric(qc["side_switch_count"], errors="coerce")
    assert_unique(qc, ["attempt_id", "target_hand", "source"], "QC table")
    return qc


def calculate_eligibility(config: dict, mapping: pd.DataFrame, qc: pd.DataFrame) -> pd.DataFrame:
    resolved = mapping[mapping["mapping_status"].isin(RESOLVED_STATUSES)].copy()
    if not resolved.empty:
        assert_unique(resolved, ["attempt_id", "target_hand", "source"], "Resolved mapping")
    merged = mapping.merge(
        qc,
        on=["attempt_id", "target_hand", "source"],
        how="left",
        suffixes=("", "_qc"),
        indicator="qc_merge",
    )
    merged["mapping_ok"] = merged["mapping_status"].isin(RESOLVED_STATUSES)
    merged["visible_window"] = merged["coverage_state"].isin(VISIBLE_STATES)
    merged["hard_qc_fail"] = merged["hard_qc"].map(hard_qc_failed)
    merged["correct_gopro_target"] = merged.apply(
        lambda row: source_target_hand(row["source"]) in {"", row["target_hand"]}, axis=1
    )
    merged["qc_present"] = merged["qc_merge"].eq("both")

    thresholds = config.get("eligibility") or {}
    primary_threshold = float(thresholds.get("primary_detection_coverage", 0.60))
    high_threshold = float(thresholds.get("high_quality_detection_coverage", 0.80))
    candidate_threshold = float(thresholds.get("candidate_detection_coverage", 0.35))
    trajectory_rules = config.get("trajectory_qc") or {}
    min_bins = float(trajectory_rules.get("min_temporal_bin_coverage", 0.70))
    max_gap = float(trajectory_rules.get("max_longest_gap_fraction", 0.20))
    min_points = int(trajectory_rules.get("min_finite_wrist_points", 30))

    base = (
        merged["mapping_ok"]
        & merged["visible_window"]
        & merged["qc_present"]
        & ~merged["hard_qc_fail"]
        & merged["correct_gopro_target"]
    )
    is_primary_source = merged.apply(
        lambda row: row["source"] == f"gopro_{row['target_hand']}", axis=1
    )
    is_cae = merged["source"].eq("cae_hand")
    coverage = merged["target_detection_coverage"]
    merged["primary_gopro60"] = base & is_primary_source & coverage.ge(primary_threshold)
    merged["highqc_gopro80"] = base & is_primary_source & coverage.ge(high_threshold)
    merged["cae_recovery60"] = base & is_cae & coverage.ge(primary_threshold)
    merged["sub60_validation_candidate"] = (
        base
        & (is_primary_source | is_cae)
        & coverage.ge(candidate_threshold)
        & coverage.lt(primary_threshold)
    )

    reported_pass = merged["trajectory_qc_pass"].map(pass_value)
    computed_pass = (
        merged["temporal_bin_coverage"].ge(min_bins)
        & merged["longest_gap_fraction"].le(max_gap)
        & merged["finite_wrist_points"].ge(min_points)
    )
    merged["trajectory_qc_basis"] = np.where(
        reported_pass.notna(), "pipeline_flag", "configured_temporal_rules"
    )
    merged["trajectory_qc_effective_pass"] = np.where(
        reported_pass.notna(), reported_pass.eq(True), computed_pass
    ).astype(bool)
    merged["primary_trajectory_eligible"] = (
        merged["primary_gopro60"] & merged["trajectory_qc_effective_pass"]
    )
    merged["cae_trajectory_recovery"] = (
        merged["cae_recovery60"] & merged["trajectory_qc_effective_pass"]
    )

    def reason(row: pd.Series) -> str:
        if not row["mapping_ok"]:
            return f"mapping_{row['mapping_status'] or 'unresolved'}"
        if row["coverage_state"] in {"camera_missing", "not_visible"}:
            return row["coverage_state"]
        if not row["correct_gopro_target"]:
            return "wrong_anatomical_target_for_gopro_side"
        if not row["qc_present"]:
            return "qc_record_missing"
        if row["hard_qc_fail"]:
            return "hard_identity_or_reconstruction_qc"
        if pd.isna(row["target_detection_coverage"]):
            return "coverage_missing"
        if row["target_detection_coverage"] < candidate_threshold:
            return "coverage_below_candidate_floor"
        if row["target_detection_coverage"] < primary_threshold:
            return "coverage_candidate_requires_validation"
        if not row["trajectory_qc_effective_pass"]:
            return "trajectory_temporal_qc"
        return "eligible"

    merged["exclusion_or_eligibility_reason"] = merged.apply(reason, axis=1)
    return merged


def select_sources(eligibility: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    keys = ["attempt_id", "participant", "configuration", "target_hand"]
    for key, group in eligibility.groupby(keys, dropna=False):
        primary = group[group["primary_trajectory_eligible"]]
        cae = group[group["cae_trajectory_recovery"]]
        if len(primary) > 1 or len(cae) > 1:
            raise ValueError(f"Multiple eligible sources for {key}; resolve mapping before selection")
        if len(primary) == 1:
            chosen = primary.iloc[0]
            analysis_set = "primary_gopro60"
        elif len(cae) == 1:
            chosen = cae.iloc[0]
            analysis_set = "cae_recovery60"
        else:
            chosen = group.iloc[0]
            analysis_set = "ineligible"
        rows.append(
            {
                "attempt_id": key[0],
                "participant": key[1],
                "configuration": key[2],
                "target_hand": key[3],
                "analysis_set": analysis_set,
                "selected_motion_source": chosen["source"] if analysis_set != "ineligible" else "",
                "selected_video_path": chosen.get("video_path", "") if analysis_set != "ineligible" else "",
                "target_detection_coverage": chosen.get("target_detection_coverage", np.nan),
                "trajectory_qc_basis": chosen.get("trajectory_qc_basis", ""),
                "reason": "eligible" if analysis_set != "ineligible" else "; ".join(
                    sorted(set(group["exclusion_or_eligibility_reason"].astype(str)))
                ),
            }
        )
    return pd.DataFrame(rows)


def count_summary(eligibility: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for flag in [
        "primary_gopro60",
        "highqc_gopro80",
        "primary_trajectory_eligible",
        "cae_recovery60",
        "cae_trajectory_recovery",
        "sub60_validation_candidate",
    ]:
        subset = eligibility[eligibility[flag]]
        for hand in ("left", "right"):
            hand_subset = subset[subset["target_hand"].eq(hand)]
            rows.append(
                {
                    "analysis_set": flag,
                    "target_hand": hand,
                    "attempts": hand_subset["attempt_id"].nunique(),
                    "participants": hand_subset["participant"].nunique(),
                }
            )
    for analysis_set in ["primary_gopro60", "cae_recovery60", "ineligible"]:
        subset = selected[selected["analysis_set"].eq(analysis_set)]
        for hand in ("left", "right"):
            hand_subset = subset[subset["target_hand"].eq(hand)]
            rows.append(
                {
                    "analysis_set": f"selected_{analysis_set}",
                    "target_hand": hand,
                    "attempts": hand_subset["attempt_id"].nunique(),
                    "participants": hand_subset["participant"].nunique(),
                }
            )
    return pd.DataFrame(rows)


def nested_checks(counts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for hand in ("left", "right"):
        primary = counts[(counts.analysis_set == "primary_gopro60") & (counts.target_hand == hand)]
        trajectory = counts[(counts.analysis_set == "primary_trajectory_eligible") & (counts.target_hand == hand)]
        if primary.empty or trajectory.empty:
            continue
        pa, pp = int(primary.iloc[0].attempts), int(primary.iloc[0].participants)
        ta, tp = int(trajectory.iloc[0].attempts), int(trajectory.iloc[0].participants)
        attempt_loss, participant_loss = pa - ta, pp - tp
        status = "pass" if 0 <= participant_loss <= attempt_loss else "fail"
        rows.append(
            {
                "check": f"nested_participant_logic_{hand}",
                "status": status,
                "detail": f"primary {pa}/{pp}; trajectory {ta}/{tp}; lost {attempt_loss} attempts and {participant_loss} participants",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit GoPro primary and CAE recovery eligibility.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mapping", default="")
    parser.add_argument("--qc", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    recovery_root = ensure_recovery_root(config)
    mapping_path = (
        Path(args.mapping).expanduser().resolve()
        if args.mapping
        else recovery_root / "02_mapping_review" / "mapping_review_queue.csv"
    )
    qc_path = (
        Path(args.qc).expanduser().resolve()
        if args.qc
        else resolve_path(config, config.get("attempt_hand_qc"))
    )
    if qc_path is None or not qc_path.exists():
        raise FileNotFoundError("Provide --qc or config.attempt_hand_qc")

    mapping = load_mapping(config, mapping_path)
    qc = load_qc(config, qc_path)
    eligibility = calculate_eligibility(config, mapping, qc)
    selected = select_sources(eligibility)
    counts = count_summary(eligibility, selected)
    checks = nested_checks(counts)

    output_dir = recovery_root / "03_eligibility_audit"
    write_csv(eligibility, output_dir / "attempt_hand_source_eligibility.csv")
    write_csv(selected, output_dir / "analysis_source_selection.csv")
    write_csv(counts, output_dir / "eligibility_counts.csv")
    write_csv(checks, output_dir / "nested_count_checks.csv")
    flow = (
        eligibility.groupby(["source", "target_hand", "exclusion_or_eligibility_reason"], dropna=False)
        .agg(rows=("attempt_id", "size"), attempts=("attempt_id", "nunique"), participants=("participant", "nunique"))
        .reset_index()
    )
    write_csv(flow, output_dir / "exclusion_flow.csv")
    write_csv(
        eligibility[eligibility["sub60_validation_candidate"]].copy(),
        output_dir / "sub60_validation_candidates.csv",
    )
    print(f"Wrote eligibility audit to {output_dir}")
    if not checks.empty and checks["status"].eq("fail").any():
        print("WARNING: nested sample/participant counts are internally inconsistent; inspect nested_count_checks.csv")


if __name__ == "__main__":
    main()
