from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd


GOOD_BOOTSTRAP = {
    "single_file_midpoint_bootstrap",
    "cross_file_midpoint_bootstrap",
}

NO_BOOTSTRAP = {
    "no_bootstrap_fit",
    "midpoint_outside_group",
}

FINAL_QUEUE_COLUMNS = [
    "queue_id",
    "priority",
    "participant",
    "block_id",
    "block_index",
    "first_attempt_id",
    "last_attempt_id",
    "attempt_count",
    "source",
    "target_hands",
    "source_role",
    "review_type",
    "recommended_group_status",
    "sequence_mode",
    "source_file_count",
    "source_files",
    "recommended_video_ids",
    "recommended_video_files",
    "recommended_video_paths",
    "block_span_sec",
    "reference_min_start_sec",
    "reference_max_end_sec",
    "offset_min_sec",
    "offset_max_sec",
    "offset_mid_sec",
    "offset_uncertainty_sec",
    "structural_precision",
    "fit_slack_sec",
    "bootstrap_attempt_count",
    "bootstrap_single_file_count",
    "bootstrap_cross_file_count",
    "bootstrap_no_fit_count",
    "anchor_attempt_id",
    "anchor_reference_start_sec",
    "anchor_video_file",
    "anchor_video_id",
    "anchor_bootstrap_local_sec",
    "anchor_bootstrap_virtual_sec",
    "review_seek_start_sec",
    "validation_status",
    "validated_anchor_local_sec",
    "validated_offset_sec",
    "validated_block_start_local_sec",
    "validated_block_end_local_sec",
    "reviewer",
    "review_note",
]


def participant_number(value: object) -> int:
    m = re.search(r"(\d+)", str(value))
    return int(m.group(1)) if m else 999999


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str).fillna("")


def to_float(value, default=float("nan")) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def to_int(value, default=0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def source_hand(source: str) -> str:
    if source == "gopro_left":
        return "left"
    if source == "gopro_right":
        return "right"
    return ""


def plan_lookup(plan: pd.DataFrame) -> dict[tuple[str, str, str], pd.Series]:
    result = {}
    for _, row in plan.iterrows():
        result[(row["participant"], row["block_id"], row["source"])] = row
    return result


def bootstrap_group(
    bootstrap: pd.DataFrame,
    participant: str,
    block_id: str,
    source: str,
) -> pd.DataFrame:
    return bootstrap[
        bootstrap["participant"].eq(participant)
        & bootstrap["block_id"].eq(block_id)
        & bootstrap["source"].eq(source)
    ].copy()


def viable_bootstrap(group: pd.DataFrame) -> bool:
    if group.empty:
        return False
    return group["bootstrap_status"].isin(GOOD_BOOTSTRAP).any()


def choose_anchor(group: pd.DataFrame) -> dict:
    """
    Prefer a single-file attempt near the middle of the block.
    This usually makes visual review easier than an edge/cross-file case.
    """
    if group.empty:
        return {
            "anchor_attempt_id": "",
            "anchor_reference_start_sec": "",
            "anchor_video_file": "",
            "anchor_video_id": "",
            "anchor_bootstrap_local_sec": "",
            "anchor_bootstrap_virtual_sec": "",
            "review_seek_start_sec": "",
        }

    g = group.copy()
    g["attempt_sequence_num"] = pd.to_numeric(
        g["attempt_sequence"], errors="coerce"
    )

    good = g[g["bootstrap_status"].isin(GOOD_BOOTSTRAP)].copy()
    if good.empty:
        good = g.copy()

    single = good[
        good["bootstrap_status"].eq("single_file_midpoint_bootstrap")
    ].copy()

    pool = single if not single.empty else good

    seq = pd.to_numeric(pool["attempt_sequence"], errors="coerce")
    median_seq = seq.median()
    pool["_distance"] = (seq - median_seq).abs()
    pool = pool.sort_values(
        ["_distance", "attempt_sequence_num"]
    )

    row = pool.iloc[0]

    local = to_float(row.get("bootstrap_start_local_sec", ""))
    seek = max(local - 20.0, 0.0) if math.isfinite(local) else ""

    return {
        "anchor_attempt_id": row.get("attempt_id", ""),
        "anchor_reference_start_sec": row.get("reference_start_sec", ""),
        "anchor_video_file": row.get("bootstrap_start_video_file", ""),
        "anchor_video_id": row.get("bootstrap_start_video_id", ""),
        "anchor_bootstrap_local_sec": row.get(
            "bootstrap_start_local_sec", ""
        ),
        "anchor_bootstrap_virtual_sec": row.get(
            "bootstrap_virtual_start_sec", ""
        ),
        "review_seek_start_sec": seek,
    }


def common_queue_fields(
    participant: str,
    block_id: str,
    source: str,
    plan_row: pd.Series,
    group: pd.DataFrame,
) -> dict:
    attempts = (
        group[["attempt_id", "attempt_sequence"]]
        .drop_duplicates()
        .copy()
    )
    attempts["seq"] = pd.to_numeric(
        attempts["attempt_sequence"], errors="coerce"
    )
    attempts = attempts.sort_values("seq")

    statuses = group["bootstrap_status"].astype(str)

    first_attempt = (
        attempts.iloc[0]["attempt_id"] if len(attempts) else ""
    )
    last_attempt = (
        attempts.iloc[-1]["attempt_id"] if len(attempts) else ""
    )

    ref_start = pd.to_numeric(
        group["reference_start_sec"], errors="coerce"
    )
    ref_end = pd.to_numeric(
        group["reference_end_sec"], errors="coerce"
    )

    result = {
        "participant": participant,
        "block_id": block_id,
        "block_index": plan_row.get("block_index", ""),
        "first_attempt_id": first_attempt,
        "last_attempt_id": last_attempt,
        "attempt_count": len(attempts),
        "source": source,
        "recommended_group_status": plan_row.get(
            "recommended_group_status", ""
        ),
        "sequence_mode": plan_row.get("sequence_mode", ""),
        "source_file_count": plan_row.get("source_file_count", ""),
        "source_files": plan_row.get("source_files", ""),
        "recommended_video_ids": plan_row.get(
            "recommended_video_ids", ""
        ),
        "recommended_video_files": plan_row.get(
            "recommended_video_files", ""
        ),
        "recommended_video_paths": plan_row.get(
            "recommended_video_paths", ""
        ),
        "block_span_sec": plan_row.get("block_span_sec", ""),
        "reference_min_start_sec": (
            ref_start.min() if ref_start.notna().any() else ""
        ),
        "reference_max_end_sec": (
            ref_end.max() if ref_end.notna().any() else ""
        ),
        "offset_min_sec": plan_row.get("offset_min_sec", ""),
        "offset_max_sec": plan_row.get("offset_max_sec", ""),
        "offset_mid_sec": plan_row.get("offset_mid_sec", ""),
        "offset_uncertainty_sec": plan_row.get(
            "offset_uncertainty_sec", ""
        ),
        "structural_precision": plan_row.get(
            "structural_precision", ""
        ),
        "fit_slack_sec": plan_row.get("fit_slack_sec", ""),
        "bootstrap_attempt_count": len(attempts),
        "bootstrap_single_file_count": int(
            statuses.eq("single_file_midpoint_bootstrap").sum()
        ),
        "bootstrap_cross_file_count": int(
            statuses.eq("cross_file_midpoint_bootstrap").sum()
        ),
        "bootstrap_no_fit_count": int(
            statuses.isin(NO_BOOTSTRAP).sum()
        ),
        "validation_status": "pending",
        "validated_anchor_local_sec": "",
        "validated_offset_sec": "",
        "validated_block_start_local_sec": "",
        "validated_block_end_local_sec": "",
        "reviewer": "",
        "review_note": "",
    }

    result.update(choose_anchor(group))
    return result


def build_queue(
    plan: pd.DataFrame,
    bootstrap: pd.DataFrame,
    include_all_cae: bool,
) -> pd.DataFrame:
    lookup = plan_lookup(plan)
    rows = []

    block_keys = (
        bootstrap[["participant", "block_id"]]
        .drop_duplicates()
        .copy()
    )
    block_keys["_p"] = block_keys["participant"].map(participant_number)
    block_keys["_b"] = pd.to_numeric(
        block_keys["block_id"].str.extract(r"B(\d+)")[0],
        errors="coerce",
    )
    block_keys = block_keys.sort_values(["_p", "_b"])

    for _, block in block_keys.iterrows():
        participant = block["participant"]
        block_id = block["block_id"]

        fallback_hands = []

        # Primary GoPro rows: one queue item per actual hand/camera.
        for source in ("gopro_left", "gopro_right"):
            key = (participant, block_id, source)
            if key not in lookup:
                continue

            p = lookup[key]
            group = bootstrap_group(
                bootstrap, participant, block_id, source
            )

            source_count = to_int(p.get("source_file_count", 0))
            status = str(p.get("recommended_group_status", ""))

            if source_count <= 0 or status == "no_source_video":
                fallback_hands.append(source_hand(source))
                continue

            base = common_queue_fields(
                participant, block_id, source, p, group
            )

            if viable_bootstrap(group):
                base["priority"] = 2
                base["target_hands"] = source_hand(source)
                base["source_role"] = "primary"
                base["review_type"] = "validate_primary_block_offset"
                rows.append(base)
            else:
                # Some GoPro exists but cannot cover/fit the complete
                # reference block. Keep it in the queue as a fragment review
                # so potentially valid primary footage is not discarded.
                base["priority"] = 1
                base["target_hands"] = source_hand(source)
                base["source_role"] = "primary_fragment"
                base["review_type"] = "identify_primary_fragment_coverage"
                base["validation_status"] = "needs_fragment_review"
                if not base["review_note"]:
                    base["review_note"] = (
                        "GoPro files exist but no complete block bootstrap "
                        "fit was possible. Identify which attempts, if any, "
                        "are genuinely present before using CAE recovery."
                    )
                rows.append(base)
                fallback_hands.append(source_hand(source))

        # CAE-HAND is queued once per block, not once per hand. It is added
        # when at least one primary hand needs recovery, or explicitly when
        # --include-all-cae is requested.
        cae_key = (participant, block_id, "cae_hand")
        if cae_key in lookup:
            p = lookup[cae_key]
            group = bootstrap_group(
                bootstrap, participant, block_id, "cae_hand"
            )
            source_count = to_int(p.get("source_file_count", 0))
            status = str(p.get("recommended_group_status", ""))

            need_cae = bool(fallback_hands) or include_all_cae

            if (
                need_cae
                and source_count > 0
                and status != "no_source_video"
            ):
                base = common_queue_fields(
                    participant, block_id, "cae_hand", p, group
                )
                base["priority"] = 1 if fallback_hands else 3
                base["target_hands"] = (
                    ";".join(sorted(set(fallback_hands)))
                    if fallback_hands
                    else "left;right"
                )
                base["source_role"] = (
                    "secondary_recovery"
                    if fallback_hands
                    else "secondary_validation"
                )
                base["review_type"] = (
                    "validate_cae_recovery_offset"
                    if viable_bootstrap(group)
                    else "cae_manual_timeline_review"
                )
                if not viable_bootstrap(group):
                    base["validation_status"] = "needs_manual_timeline_review"
                rows.append(base)

    queue = pd.DataFrame(rows)

    if queue.empty:
        return pd.DataFrame(columns=FINAL_QUEUE_COLUMNS)

    queue["_p"] = queue["participant"].map(participant_number)
    queue["_b"] = pd.to_numeric(
        queue["block_index"], errors="coerce"
    )
    source_order = {
        "gopro_left": 0,
        "gopro_right": 1,
        "cae_hand": 2,
    }
    queue["_s"] = queue["source"].map(source_order).fillna(99)

    queue = queue.sort_values(
        ["priority", "_p", "_b", "_s"]
    ).drop(columns=["_p", "_b", "_s"]).reset_index(drop=True)

    queue.insert(
        0,
        "queue_id",
        [
            f"Q{i:03d}"
            for i in range(1, len(queue) + 1)
        ],
    )

    for col in FINAL_QUEUE_COLUMNS:
        if col not in queue.columns:
            queue[col] = ""

    return queue[FINAL_QUEUE_COLUMNS]


def write_summary(queue: pd.DataFrame, output_dir: Path):
    summary_path = output_dir / "block_validation_queue_summary.csv"

    if queue.empty:
        pd.DataFrame(
            columns=["source", "source_role", "review_type", "count"]
        ).to_csv(summary_path, index=False)
        return summary_path

    summary = (
        queue.groupby(
            ["source", "source_role", "review_type"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values(["source", "source_role", "review_type"])
    )
    summary.to_csv(summary_path, index=False)
    return summary_path


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Reduce attempt_windows_bootstrap.csv to a block-level visual "
            "validation queue. GoPro is primary; CAE-HAND is queued once per "
            "block only where primary coverage needs recovery, unless "
            "--include-all-cae is used."
        )
    )
    ap.add_argument("--bootstrap", required=True)
    ap.add_argument("--timing-plan", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--include-all-cae",
        action="store_true",
        help=(
            "Also queue CAE-HAND where both primary GoPro sides already have "
            "usable block bootstraps. Default is recovery-only CAE."
        ),
    )
    args = ap.parse_args()

    bootstrap = read_csv(Path(args.bootstrap))
    plan = read_csv(Path(args.timing_plan))

    required_bootstrap = {
        "participant","block_id","source","attempt_id","attempt_sequence",
        "reference_start_sec","reference_end_sec","bootstrap_status",
        "bootstrap_start_video_file","bootstrap_start_video_id",
        "bootstrap_start_local_sec","bootstrap_virtual_start_sec",
    }
    missing = required_bootstrap - set(bootstrap.columns)
    if missing:
        raise RuntimeError(
            "attempt_windows_bootstrap.csv is missing columns: "
            + ", ".join(sorted(missing))
        )

    required_plan = {
        "participant","block_id","block_index","source",
        "recommended_group_status","source_file_count","source_files",
        "block_span_sec",
    }
    missing = required_plan - set(plan.columns)
    if missing:
        raise RuntimeError(
            "timing_block_plan.csv is missing columns: "
            + ", ".join(sorted(missing))
        )

    queue = build_queue(
        plan,
        bootstrap,
        include_all_cae=args.include_all_cae,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    queue_path = output_dir / "block_validation_queue.csv"
    queue.to_csv(queue_path, index=False)
    summary_path = write_summary(queue, output_dir)

    print("=== BLOCK VALIDATION QUEUE CREATED ===")
    print(f"Queue rows: {len(queue)}")
    print(f"Wrote: {queue_path}")
    print(f"Wrote: {summary_path}")
    print()

    if len(queue):
        print("By source / role / review type:")
        print(
            queue.groupby(
                ["source","source_role","review_type"]
            ).size().to_string()
        )
        print()

        print("Priority counts:")
        print(queue["priority"].value_counts().sort_index().to_string())
        print()

        print("First 20 queue items:")
        show = [
            "queue_id","priority","participant","block_id",
            "source","target_hands","source_role","review_type",
            "first_attempt_id","last_attempt_id",
            "anchor_attempt_id","anchor_video_file",
            "anchor_bootstrap_local_sec","offset_uncertainty_sec",
            "validation_status",
        ]
        print(queue[show].head(20).to_string(index=False))

    print()
    print("VALIDATION FIELDS TO EDIT LATER:")
    print("  validation_status")
    print("  validated_anchor_local_sec")
    print("  validated_offset_sec")
    print("  validated_block_start_local_sec")
    print("  validated_block_end_local_sec")
    print("  reviewer")
    print("  review_note")
    print()
    print(
        "Do not run final hand metrics from this queue. This queue is the "
        "small visual-review layer used to validate/refine block timing first."
    )


if __name__ == "__main__":
    main()
