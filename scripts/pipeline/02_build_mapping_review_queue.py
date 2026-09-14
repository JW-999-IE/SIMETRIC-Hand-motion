from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from common import (
    ATTEMPT_ALIASES,
    MAPPING_ALIASES,
    canonicalize_columns,
    ensure_recovery_root,
    load_config,
    normalize_participant,
    normalize_source,
    parse_bool,
    read_table,
    resolve_path,
    write_csv,
)


def review_id(attempt_id: str, target_hand: str, source: str, video_path: str) -> str:
    payload = f"{attempt_id}|{target_hand}|{source}|{Path(video_path).name}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def load_attempts(config: dict) -> pd.DataFrame:
    path = resolve_path(config, config.get("attempt_workbook"))
    if path is None or not path.exists():
        raise FileNotFoundError("attempt_workbook is missing")
    sheet = config.get("attempt_sheet", 0)
    attempts = read_table(path, sheet_name=sheet)
    attempts = canonicalize_columns(
        attempts,
        (config.get("column_maps") or {}).get("attempts"),
        ATTEMPT_ALIASES,
        required=["attempt_id", "participant", "configuration"],
    )
    attempts["attempt_id"] = attempts["attempt_id"].astype(str).str.strip()
    attempts["participant"] = attempts["participant"].map(normalize_participant)
    if "aborted" in attempts:
        attempts = attempts[~attempts["aborted"].map(parse_bool)].copy()
    return attempts


def load_existing_mapping(config: dict) -> pd.DataFrame:
    path = resolve_path(config, config.get("frozen_attempt_mapping"))
    if path is None or not path.exists():
        return pd.DataFrame()
    mapping = read_table(path)
    mapping = canonicalize_columns(
        mapping,
        (config.get("column_maps") or {}).get("mapping"),
        MAPPING_ALIASES,
        required=["attempt_id", "source", "video_path"],
    )
    mapping["attempt_id"] = mapping["attempt_id"].astype(str).str.strip()
    mapping["source"] = mapping["source"].map(normalize_source)
    mapping["video_name_key"] = mapping["video_path"].astype(str).map(lambda x: Path(x).name.lower())
    if "mapping_status" not in mapping:
        mapping["mapping_status"] = ""
    return mapping


def build_queue(config: dict, attempts: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    inventory = inventory[inventory["inventory_status"].eq("ok")].copy()
    inventory["participant"] = inventory["participant"].map(normalize_participant)
    inventory["source"] = inventory["source"].map(normalize_source)
    existing = load_existing_mapping(config)

    rows: list[dict] = []
    for attempt in attempts.to_dict("records"):
        for target_hand, primary_source in (("left", "gopro_left"), ("right", "gopro_right")):
            for source, source_role in ((primary_source, "primary"), ("cae_hand", "recovery")):
                candidates = inventory[
                    inventory["participant"].eq(attempt["participant"])
                    & inventory["source"].eq(source)
                ]
                if candidates.empty:
                    rows.append(
                        {
                            "review_id": review_id(attempt["attempt_id"], target_hand, source, "missing"),
                            "attempt_id": attempt["attempt_id"],
                            "participant": attempt["participant"],
                            "configuration": attempt["configuration"],
                            "target_hand": target_hand,
                            "source": source,
                            "source_role": source_role,
                            "video_path": "",
                            "video_name": "",
                            "video_duration_sec": "",
                            "mapping_status": "camera_missing",
                            "coverage_state": "camera_missing",
                            "video_local_start_sec": "",
                            "video_local_end_sec": "",
                            "mapping_confidence": "",
                            "review_notes": "No candidate recording found; do not infer from the opposite GoPro camera.",
                        }
                    )
                    continue

                for candidate in candidates.to_dict("records"):
                    mapping_status = "needs_video_review"
                    coverage_state = ""
                    start_sec = ""
                    end_sec = ""
                    confidence = ""
                    notes = ""
                    if not existing.empty:
                        match = existing[
                            existing["attempt_id"].eq(attempt["attempt_id"])
                            & existing["source"].eq(source)
                            & existing["video_name_key"].eq(str(candidate["video_name"]).lower())
                        ]
                        if len(match) == 1:
                            record = match.iloc[0]
                            mapping_status = str(record.get("mapping_status", "")).strip() or "existing_mapping_review"
                            coverage_state = str(record.get("coverage_state", "")).strip()
                            start_sec = record.get("video_local_start_sec", "")
                            end_sec = record.get("video_local_end_sec", "")
                            confidence = record.get("mapping_confidence", "")
                            notes = str(record.get("review_notes", "")).strip()

                    rows.append(
                        {
                            "review_id": review_id(attempt["attempt_id"], target_hand, source, candidate["video_path"]),
                            "attempt_id": attempt["attempt_id"],
                            "participant": attempt["participant"],
                            "configuration": attempt["configuration"],
                            "target_hand": target_hand,
                            "source": source,
                            "source_role": source_role,
                            "video_path": candidate["video_path"],
                            "video_name": candidate["video_name"],
                            "video_duration_sec": candidate.get("duration_sec", ""),
                            "mapping_status": mapping_status,
                            "coverage_state": coverage_state,
                            "video_local_start_sec": start_sec,
                            "video_local_end_sec": end_sec,
                            "mapping_confidence": confidence,
                            "review_notes": notes,
                        }
                    )
    queue = pd.DataFrame(rows)
    return queue.sort_values(["participant", "attempt_id", "target_hand", "source", "video_name"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a manual attempt-to-video review queue without guessing mappings."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--inventory", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    recovery_root = ensure_recovery_root(config)
    inventory_path = (
        Path(args.inventory).expanduser().resolve()
        if args.inventory
        else recovery_root / "01_inventory" / "video_inventory_recheck.csv"
    )
    attempts = load_attempts(config)
    inventory = read_table(inventory_path)
    queue = build_queue(config, attempts, inventory)
    output = recovery_root / "02_mapping_review" / "mapping_review_queue.csv"
    write_csv(queue, output)

    summary = (
        queue.groupby(["source", "target_hand", "mapping_status"], dropna=False)
        .agg(rows=("review_id", "count"), attempts=("attempt_id", "nunique"), participants=("participant", "nunique"))
        .reset_index()
    )
    write_csv(summary, output.parent / "mapping_queue_summary.csv")
    print(f"Wrote {len(queue)} review rows to {output}")
    print("Complete video-local start/end times and mapping decisions from actual video evidence.")


if __name__ == "__main__":
    main()
