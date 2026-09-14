from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_EXCLUDED = {"P7", "P8", "P14"}


def natural_key(value: object):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", str(value))
    ]


def participant_number(value: object) -> int:
    m = re.search(r"(\d+)", str(value))
    return int(m.group(1)) if m else 999999


def normalize_participant(value: object) -> str:
    m = re.search(r"(\d+)", str(value))
    return f"P{int(m.group(1))}" if m else str(value)


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str).fillna("")


def num(value, default=float("nan")) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def prepare_attempts(mapping_v2: pd.DataFrame, excluded: set[str], reset_tolerance_ms: float):
    """
    Derive one participant-level attempt timeline from the v2 map.
    Left/right rows carry the same reference attempt clock, so deduplicate
    by attempt_id before detecting clock-reset blocks.
    """
    needed = {
        "attempt_id", "participant", "attempt_sequence",
        "round", "device", "reference_start_ms", "reference_end_ms",
    }
    missing = needed - set(mapping_v2.columns)
    if missing:
        raise RuntimeError(
            "attempt_video_map_v2.csv is missing required columns: "
            + ", ".join(sorted(missing))
        )

    base = (
        mapping_v2[list(needed)]
        .drop_duplicates("attempt_id")
        .copy()
    )

    base["participant"] = base["participant"].map(normalize_participant)
    base = base[~base["participant"].isin(excluded)].copy()

    base["attempt_sequence_num"] = pd.to_numeric(
        base["attempt_sequence"], errors="coerce"
    )
    base["reference_start_ms_num"] = pd.to_numeric(
        base["reference_start_ms"], errors="coerce"
    )
    base["reference_end_ms_num"] = pd.to_numeric(
        base["reference_end_ms"], errors="coerce"
    )

    bad = base[
        base[
            ["attempt_sequence_num", "reference_start_ms_num", "reference_end_ms_num"]
        ].isna().any(axis=1)
    ]
    if len(bad):
        raise RuntimeError(
            f"{len(bad)} attempts have non-numeric sequence/start/end values."
        )

    base["_p"] = base["participant"].map(participant_number)
    base = base.sort_values(
        ["_p", "attempt_sequence_num", "attempt_id"]
    ).drop(columns="_p")

    block_rows = []
    attempt_rows = []

    for participant, group in base.groupby("participant", sort=False):
        group = group.sort_values("attempt_sequence_num").copy()

        block_index = 1
        previous_start = None
        assigned = []

        for _, row in group.iterrows():
            start_ms = float(row["reference_start_ms_num"])

            if (
                previous_start is not None
                and start_ms < previous_start - reset_tolerance_ms
            ):
                block_index += 1

            assigned.append(block_index)
            previous_start = start_ms

        group["block_index"] = assigned

        for block_index, block in group.groupby("block_index", sort=True):
            block = block.sort_values("attempt_sequence_num")
            ref_min_start_ms = float(block["reference_start_ms_num"].min())
            ref_max_end_ms = float(block["reference_end_ms_num"].max())
            span_sec = (ref_max_end_ms - ref_min_start_ms) / 1000.0

            block_id = f"{participant}_B{int(block_index):02d}"

            block_rows.append({
                "participant": participant,
                "block_id": block_id,
                "block_index": int(block_index),
                "first_attempt_id": block.iloc[0]["attempt_id"],
                "last_attempt_id": block.iloc[-1]["attempt_id"],
                "first_attempt_sequence": int(block.iloc[0]["attempt_sequence_num"]),
                "last_attempt_sequence": int(block.iloc[-1]["attempt_sequence_num"]),
                "attempt_count": len(block),
                "reference_min_start_ms": ref_min_start_ms,
                "reference_max_end_ms": ref_max_end_ms,
                "block_span_sec": span_sec,
                "rounds": ";".join(
                    dict.fromkeys(block["round"].astype(str).tolist())
                ),
                "devices": ";".join(
                    dict.fromkeys(block["device"].astype(str).tolist())
                ),
            })

            for _, a in block.iterrows():
                attempt_rows.append({
                    "participant": participant,
                    "block_id": block_id,
                    "block_index": int(block_index),
                    "attempt_id": a["attempt_id"],
                    "attempt_sequence": int(a["attempt_sequence_num"]),
                    "round": a["round"],
                    "device": a["device"],
                    "reference_start_ms": float(a["reference_start_ms_num"]),
                    "reference_end_ms": float(a["reference_end_ms_num"]),
                })

    blocks = pd.DataFrame(block_rows)
    attempts = pd.DataFrame(attempt_rows)

    blocks["_p"] = blocks["participant"].map(participant_number)
    blocks = blocks.sort_values(
        ["_p", "block_index"]
    ).drop(columns="_p").reset_index(drop=True)

    attempts["_p"] = attempts["participant"].map(participant_number)
    attempts = attempts.sort_values(
        ["_p", "attempt_sequence"]
    ).drop(columns="_p").reset_index(drop=True)

    return blocks, attempts


def prepare_gopro_inventory(inv: pd.DataFrame, excluded: set[str]) -> pd.DataFrame:
    needed = {
        "participant", "camera", "video_id", "file_name",
        "path", "duration_sec",
    }
    missing = needed - set(inv.columns)
    if missing:
        raise RuntimeError(
            "video_inventory.csv is missing required columns: "
            + ", ".join(sorted(missing))
        )

    out = inv.copy()
    out["participant"] = out["participant"].map(normalize_participant)
    out = out[~out["participant"].isin(excluded)].copy()

    out["camera"] = out["camera"].astype(str).str.lower()

    if "probe_status" in out.columns:
        out = out[out["probe_status"].astype(str).eq("ok")]

    if "excluded" in out.columns:
        out = out[~out["excluded"].map(truthy)]

    out = out[out["camera"].isin(["left", "right"])].copy()
    out["duration_sec_num"] = pd.to_numeric(
        out["duration_sec"], errors="coerce"
    )
    out = out[out["duration_sec_num"].notna() & (out["duration_sec_num"] > 0)]

    return out


def prepare_cae_inventory(
    inv: pd.DataFrame,
    extraction_result: pd.DataFrame | None,
    excluded: set[str],
    cae_output_root: Path | None = None,
) -> pd.DataFrame:
    if inv.empty:
        return inv

    needed = {
        "participant", "camera_source", "video_id",
        "file_name", "path",
    }
    missing = needed - set(inv.columns)
    if missing:
        raise RuntimeError(
            "cae_video_inventory.csv is missing required columns: "
            + ", ".join(sorted(missing))
        )

    out = inv.copy()
    out["participant"] = out["participant"].map(normalize_participant)
    out = out[~out["participant"].isin(excluded)].copy()
    out = out[out["camera_source"].astype(str).eq("cae_hand")].copy()

    if extraction_result is not None and not extraction_result.empty:
        complete = set(
            extraction_result[
                extraction_result["status"].astype(str).eq("complete")
            ]["video_id"].astype(str)
        )
        out = out[out["video_id"].astype(str).isin(complete)]

    # Pilot inventory may have been produced without probing. Prefer its
    # duration if present; otherwise recover duration from extraction_result.
    if "duration_sec" not in out.columns:
        out["duration_sec"] = ""

    out["duration_sec_num"] = pd.to_numeric(
        out["duration_sec"], errors="coerce"
    )

    # Batch CAE inventory is deliberately allowed to remain unprobed so that
    # cloud files are not hydrated just for metadata. Once extraction has run,
    # checkpoint.json contains fps + total_frames and therefore provides the
    # authoritative processed-video duration without reopening the MP4.
    if cae_output_root is not None:
        recovered = []
        for _, row in out.iterrows():
            current = row.get("duration_sec_num")
            if pd.notna(current) and float(current) > 0:
                recovered.append(float(current))
                continue

            checkpoint = (
                cae_output_root
                / "landmarks"
                / str(row["video_id"])
                / "checkpoint.json"
            )
            duration = float("nan")
            if checkpoint.exists():
                try:
                    data = json.loads(checkpoint.read_text(encoding="utf-8"))
                    fps = num(data.get("fps"))
                    total_frames = num(data.get("total_frames"))
                    if math.isfinite(fps) and fps > 0 and math.isfinite(total_frames) and total_frames > 0:
                        duration = total_frames / fps
                except Exception:
                    pass
            recovered.append(duration)

        out["duration_sec_num"] = recovered

    if extraction_result is not None and not extraction_result.empty:
        er = extraction_result.copy()
        er["video_id"] = er["video_id"].astype(str)
        er["total_frames_num"] = pd.to_numeric(
            er.get("total_frames", ""), errors="coerce"
        )

        # fps is not present in the old extraction_result, so duration cannot
        # always be reconstructed here. If checkpoint-derived inventory has
        # no duration, leave it unknown and exclude from mechanical fitting.
        frame_lookup = dict(
            zip(er["video_id"], er["total_frames_num"])
        )
        out["extraction_total_frames"] = out["video_id"].astype(str).map(frame_lookup)
    else:
        out["extraction_total_frames"] = ""

    return out


def sort_files(group: pd.DataFrame) -> list[dict]:
    records = group.to_dict("records")
    records.sort(key=lambda r: natural_key(r["file_name"]))
    return records


def consecutive_groups(files: list[dict], max_group_files: int):
    result = []
    n = len(files)

    for start in range(n):
        total = 0.0
        members = []

        for end in range(start, min(n, start + max_group_files)):
            duration = num(files[end].get("duration_sec_num"))
            if not math.isfinite(duration) or duration <= 0:
                break

            total += duration
            members.append(files[end])

            result.append({
                "start_index": start,
                "end_index": end,
                "file_count": len(members),
                "total_duration_sec": total,
                "video_ids": [str(x["video_id"]) for x in members],
                "file_names": [str(x["file_name"]) for x in members],
                "paths": [str(x["path"]) for x in members],
                "durations_sec": [
                    float(x["duration_sec_num"]) for x in members
                ],
            })

    return result


def candidate_groups_for_block(
    block: pd.Series,
    groups: list[dict],
    fit_tolerance_sec: float,
):
    span = float(block["block_span_sec"])
    ref_min = float(block["reference_min_start_ms"]) / 1000.0
    ref_max = float(block["reference_max_end_ms"]) / 1000.0

    candidates = []

    for g in groups:
        duration = float(g["total_duration_sec"])

        if duration + fit_tolerance_sec < span:
            continue

        offset_min = ref_max - duration
        offset_max = ref_min
        width = offset_max - offset_min

        if width < -fit_tolerance_sec:
            continue

        # Clamp tiny negative width caused by tolerance/rounding.
        width = max(width, 0.0)
        slack = duration - span

        c = dict(g)
        c.update({
            "offset_min_sec": offset_min,
            "offset_max_sec": offset_max,
            "offset_width_sec": width,
            "fit_slack_sec": slack,
            # Smaller uncertainty/slack is structurally preferable.
            "cost": max(slack, 0.0) + 0.001 * g["file_count"],
        })
        candidates.append(c)

    candidates.sort(
        key=lambda c: (
            c["cost"],
            c["file_count"],
            c["start_index"],
            c["end_index"],
        )
    )
    return candidates


def sequence_assign(candidate_lists: list[list[dict]], mode: str):
    """
    Dynamic programming across ordered timing blocks.

    strict:
        each subsequent timing block must use files strictly after the
        previous group's final file.

    relaxed:
        file order cannot go backwards, but reuse/overlap is allowed with a
        large penalty. This is useful for CAE continuous recordings and
        structurally awkward GoPro cases.
    """
    if not candidate_lists or any(not x for x in candidate_lists):
        return None

    states = {}

    for i, candidate in enumerate(candidate_lists[0]):
        states[i] = (
            candidate["cost"],
            [i],
        )

    for block_pos in range(1, len(candidate_lists)):
        previous_candidates = candidate_lists[block_pos - 1]
        current_candidates = candidate_lists[block_pos]
        new_states = {}

        for curr_idx, curr in enumerate(current_candidates):
            best = None

            for prev_idx, (prev_cost, path) in states.items():
                prev = previous_candidates[prev_idx]

                if mode == "strict":
                    allowed = curr["start_index"] > prev["end_index"]
                    transition_penalty = 0.0

                elif mode == "relaxed":
                    allowed = curr["start_index"] >= prev["start_index"]
                    overlap_files = max(
                        0,
                        prev["end_index"] - curr["start_index"] + 1,
                    )
                    transition_penalty = 10000.0 * overlap_files

                else:
                    raise ValueError(mode)

                if not allowed:
                    continue

                cost = prev_cost + curr["cost"] + transition_penalty

                if best is None or cost < best[0]:
                    best = (cost, path + [curr_idx])

            if best is not None:
                new_states[curr_idx] = best

        if not new_states:
            return None

        states = new_states

    best_final = min(states.values(), key=lambda x: x[0])
    return best_final[1]


def structural_precision(width_sec: float) -> str:
    if width_sec <= 10:
        return "narrow_le_10s"
    if width_sec <= 30:
        return "moderate_10_30s"
    if width_sec <= 120:
        return "wide_30_120s"
    return "very_wide_gt_120s"


def source_rows_for_participant(
    participant: str,
    blocks: pd.DataFrame,
    source: str,
    files: list[dict],
    max_group_files: int,
    fit_tolerance_sec: float,
):
    pblocks = blocks[blocks["participant"].eq(participant)].sort_values(
        "block_index"
    )

    if not files:
        return [
            {
                **block.to_dict(),
                "source": source,
                "source_priority": (
                    "primary" if source.startswith("gopro_") else "secondary"
                ),
                "source_file_count": 0,
                "source_files": "",
                "candidate_group_count": 0,
                "recommended_group_status": "no_source_video",
                "sequence_mode": "none",
                "recommended_video_ids": "",
                "recommended_video_files": "",
                "recommended_video_paths": "",
                "recommended_file_count": 0,
                "recommended_group_duration_sec": "",
                "offset_min_sec": "",
                "offset_max_sec": "",
                "offset_mid_sec": "",
                "offset_uncertainty_sec": "",
                "structural_precision": "none",
                "fit_slack_sec": "",
                "alternative_groups_json": "[]",
                "requires_visual_validation": True,
            }
            for _, block in pblocks.iterrows()
        ]

    groups = consecutive_groups(files, max_group_files)

    candidate_lists = [
        candidate_groups_for_block(
            block,
            groups,
            fit_tolerance_sec,
        )
        for _, block in pblocks.iterrows()
    ]

    assignment = sequence_assign(candidate_lists, "strict")
    sequence_mode = "strict_nonoverlap"

    if assignment is None:
        assignment = sequence_assign(candidate_lists, "relaxed")
        sequence_mode = "relaxed_ordered"

    if assignment is None:
        # At least one block has no candidate or sequence constraints cannot
        # be satisfied. Preserve any local structural information instead of
        # forcing a global solution.
        assignment = [
            0 if candidates else None
            for candidates in candidate_lists
        ]
        sequence_mode = "independent_best"

    output = []

    for block_position, (_, block) in enumerate(pblocks.iterrows()):
        candidates = candidate_lists[block_position]
        chosen_idx = assignment[block_position]

        base = {
            **block.to_dict(),
            "source": source,
            "source_priority": (
                "primary" if source.startswith("gopro_") else "secondary"
            ),
            "source_file_count": len(files),
            "source_files": ";".join(str(f["file_name"]) for f in files),
            "candidate_group_count": len(candidates),
            "sequence_mode": sequence_mode,
            "requires_visual_validation": True,
        }

        alternatives = []
        for c in candidates[:20]:
            alternatives.append({
                "video_ids": c["video_ids"],
                "file_names": c["file_names"],
                "total_duration_sec": round(c["total_duration_sec"], 6),
                "offset_min_sec": round(c["offset_min_sec"], 6),
                "offset_max_sec": round(c["offset_max_sec"], 6),
                "offset_width_sec": round(c["offset_width_sec"], 6),
                "fit_slack_sec": round(c["fit_slack_sec"], 6),
            })

        base["alternative_groups_json"] = json.dumps(
            alternatives, ensure_ascii=False
        )

        if chosen_idx is None or not candidates:
            base.update({
                "recommended_group_status": "no_duration_fit",
                "recommended_video_ids": "",
                "recommended_video_files": "",
                "recommended_video_paths": "",
                "recommended_file_count": 0,
                "recommended_group_duration_sec": "",
                "offset_min_sec": "",
                "offset_max_sec": "",
                "offset_mid_sec": "",
                "offset_uncertainty_sec": "",
                "structural_precision": "none",
                "fit_slack_sec": "",
            })

        else:
            chosen = candidates[chosen_idx]
            width = float(chosen["offset_width_sec"])

            if len(candidates) == 1:
                status = "single_structural_candidate"
            elif sequence_mode == "strict_nonoverlap":
                status = "sequence_recommended"
            elif sequence_mode == "relaxed_ordered":
                status = "relaxed_sequence_recommended"
            else:
                status = "independent_best_only"

            base.update({
                "recommended_group_status": status,
                "recommended_video_ids": ";".join(chosen["video_ids"]),
                "recommended_video_files": ";".join(chosen["file_names"]),
                "recommended_video_paths": ";".join(chosen["paths"]),
                "recommended_file_count": chosen["file_count"],
                "recommended_group_duration_sec": chosen["total_duration_sec"],
                "offset_min_sec": chosen["offset_min_sec"],
                "offset_max_sec": chosen["offset_max_sec"],
                "offset_mid_sec": (
                    chosen["offset_min_sec"] + chosen["offset_max_sec"]
                ) / 2.0,
                "offset_uncertainty_sec": width,
                "structural_precision": structural_precision(width),
                "fit_slack_sec": chosen["fit_slack_sec"],
            })

        output.append(base)

    return output


def group_virtual_segments(plan_row: pd.Series):
    files = str(plan_row["recommended_video_files"]).split(";")
    ids = str(plan_row["recommended_video_ids"]).split(";")
    paths = str(plan_row["recommended_video_paths"]).split(";")

    alternatives = json.loads(
        plan_row.get("alternative_groups_json", "[]") or "[]"
    )

    # Find the chosen group in alternatives to recover individual durations.
    chosen = None
    chosen_files = files

    for alt in alternatives:
        if alt.get("file_names") == chosen_files:
            chosen = alt
            break

    # alternative JSON intentionally does not store per-file durations, so
    # caller supplies durations from source inventory via lookup instead.
    return files, ids, paths


def locate_virtual_time(time_sec: float, files: list[dict]):
    cumulative = 0.0

    for i, file in enumerate(files):
        duration = float(file["duration_sec_num"])
        end = cumulative + duration

        # Final endpoint is inclusive for the final file.
        if time_sec < end or (i == len(files) - 1 and time_sec <= end):
            return {
                "file_index": i,
                "video_id": str(file["video_id"]),
                "file_name": str(file["file_name"]),
                "path": str(file["path"]),
                "file_local_sec": time_sec - cumulative,
            }

        cumulative = end

    return None


def bootstrap_rows(
    attempts: pd.DataFrame,
    timing_plan: pd.DataFrame,
    source_inventory_lookup: dict[tuple[str, str], list[dict]],
):
    rows = []

    for _, plan in timing_plan.iterrows():
        participant = plan["participant"]
        block_id = plan["block_id"]
        source = plan["source"]

        battempts = attempts[
            attempts["participant"].eq(participant)
            & attempts["block_id"].eq(block_id)
        ].sort_values("attempt_sequence")

        video_ids = [
            x for x in str(plan["recommended_video_ids"]).split(";") if x
        ]

        files_all = source_inventory_lookup.get(
            (participant, source), []
        )
        file_by_id = {
            str(x["video_id"]): x
            for x in files_all
        }
        chosen_files = [
            file_by_id[x]
            for x in video_ids
            if x in file_by_id
        ]

        has_fit = (
            len(chosen_files) == len(video_ids)
            and len(video_ids) > 0
            and str(plan["recommended_group_status"])
            not in {"no_source_video", "no_duration_fit"}
        )

        offset_min = num(plan["offset_min_sec"])
        offset_max = num(plan["offset_max_sec"])
        offset_mid = num(plan["offset_mid_sec"])

        for _, attempt in battempts.iterrows():
            ref_start = float(attempt["reference_start_ms"]) / 1000.0
            ref_end = float(attempt["reference_end_ms"]) / 1000.0

            base = {
                "participant": participant,
                "block_id": block_id,
                "block_index": plan["block_index"],
                "attempt_id": attempt["attempt_id"],
                "attempt_sequence": attempt["attempt_sequence"],
                "round": attempt["round"],
                "device": attempt["device"],
                "source": source,
                "target_hand": (
                    "left" if source == "gopro_left"
                    else "right" if source == "gopro_right"
                    else "both"
                ),
                "source_priority": plan["source_priority"],
                "reference_start_sec": ref_start,
                "reference_end_sec": ref_end,
                "recommended_group_status": plan["recommended_group_status"],
                "sequence_mode": plan["sequence_mode"],
                "recommended_video_ids": plan["recommended_video_ids"],
                "recommended_video_files": plan["recommended_video_files"],
                "offset_min_sec": plan["offset_min_sec"],
                "offset_max_sec": plan["offset_max_sec"],
                "offset_mid_sec": plan["offset_mid_sec"],
                "offset_uncertainty_sec": plan["offset_uncertainty_sec"],
                "structural_precision": plan["structural_precision"],
                "requires_visual_validation": True,
            }

            if not has_fit or not all(
                math.isfinite(x)
                for x in [offset_min, offset_max, offset_mid]
            ):
                base.update({
                    "bootstrap_status": "no_bootstrap_fit",
                    "virtual_start_min_sec": "",
                    "virtual_start_max_sec": "",
                    "virtual_end_min_sec": "",
                    "virtual_end_max_sec": "",
                    "bootstrap_virtual_start_sec": "",
                    "bootstrap_virtual_end_sec": "",
                    "bootstrap_start_video_id": "",
                    "bootstrap_start_video_file": "",
                    "bootstrap_start_local_sec": "",
                    "bootstrap_end_video_id": "",
                    "bootstrap_end_video_file": "",
                    "bootstrap_end_local_sec": "",
                    "bootstrap_single_file": False,
                })
                rows.append(base)
                continue

            # local = reference - offset.
            v_start_min = ref_start - offset_max
            v_start_max = ref_start - offset_min
            v_end_min = ref_end - offset_max
            v_end_max = ref_end - offset_min

            v_start_mid = ref_start - offset_mid
            v_end_mid = ref_end - offset_mid

            start_loc = locate_virtual_time(v_start_mid, chosen_files)
            end_loc = locate_virtual_time(v_end_mid, chosen_files)

            if start_loc is None or end_loc is None:
                status = "midpoint_outside_group"
                single_file = False
            elif start_loc["video_id"] == end_loc["video_id"]:
                status = "single_file_midpoint_bootstrap"
                single_file = True
            else:
                status = "cross_file_midpoint_bootstrap"
                single_file = False

            base.update({
                "bootstrap_status": status,
                "virtual_start_min_sec": v_start_min,
                "virtual_start_max_sec": v_start_max,
                "virtual_end_min_sec": v_end_min,
                "virtual_end_max_sec": v_end_max,
                "bootstrap_virtual_start_sec": v_start_mid,
                "bootstrap_virtual_end_sec": v_end_mid,
                "bootstrap_start_video_id": (
                    start_loc["video_id"] if start_loc else ""
                ),
                "bootstrap_start_video_file": (
                    start_loc["file_name"] if start_loc else ""
                ),
                "bootstrap_start_local_sec": (
                    start_loc["file_local_sec"] if start_loc else ""
                ),
                "bootstrap_end_video_id": (
                    end_loc["video_id"] if end_loc else ""
                ),
                "bootstrap_end_video_file": (
                    end_loc["file_name"] if end_loc else ""
                ),
                "bootstrap_end_local_sec": (
                    end_loc["file_local_sec"] if end_loc else ""
                ),
                "bootstrap_single_file": single_file,
            })
            rows.append(base)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["_p"] = out["participant"].map(participant_number)
        source_order = {
            "gopro_left": 0,
            "gopro_right": 1,
            "cae_hand": 2,
        }
        out["_s"] = out["source"].map(source_order).fillna(99)
        out = out.sort_values(
            ["_p", "attempt_sequence", "_s"]
        ).drop(columns=["_p", "_s"]).reset_index(drop=True)

    return out


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Build block-level recording/source plans and attempt-window "
            "bootstrap ranges without modifying the current source-aware map."
        )
    )
    ap.add_argument("--mapping-v2", required=True)
    ap.add_argument("--gopro-inventory", required=True)
    ap.add_argument("--cae-inventory")
    ap.add_argument("--cae-extraction-result")
    ap.add_argument(
        "--cae-output-root",
        help="CAE sidecar output root; used to recover fps/frame_count from checkpoints when inventory was not probed.",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--max-group-files",
        type=int,
        default=8,
        help="Maximum consecutive GoPro files allowed in one timing-block group.",
    )
    ap.add_argument(
        "--fit-tolerance-sec",
        type=float,
        default=1.0,
        help="Small tolerance when comparing block span with video-group duration.",
    )
    ap.add_argument(
        "--reset-tolerance-ms",
        type=float,
        default=0.0,
        help="A timing block resets when current start_ms is lower than the previous by more than this value.",
    )
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=["P7", "P8", "P14"],
    )
    args = ap.parse_args()

    excluded = {
        normalize_participant(x)
        for x in args.exclude
    }

    mapping_v2 = read_csv(Path(args.mapping_v2))
    gopro_inventory = prepare_gopro_inventory(
        read_csv(Path(args.gopro_inventory)),
        excluded,
    )

    cae_inventory = pd.DataFrame()
    extraction_result = None

    if args.cae_extraction_result:
        p = Path(args.cae_extraction_result)
        if p.exists():
            extraction_result = read_csv(p)

    if args.cae_inventory:
        p = Path(args.cae_inventory)
        if p.exists():
            cae_inventory = prepare_cae_inventory(
                read_csv(p),
                extraction_result,
                excluded,
                Path(args.cae_output_root) if args.cae_output_root else None,
            )

    blocks, attempts = prepare_attempts(
        mapping_v2,
        excluded,
        args.reset_tolerance_ms,
    )

    # Build source inventories separately.
    source_inventory_lookup = {}

    for participant in blocks["participant"].unique():
        for camera in ("left", "right"):
            g = gopro_inventory[
                gopro_inventory["participant"].eq(participant)
                & gopro_inventory["camera"].eq(camera)
            ]
            source_inventory_lookup[
                (participant, f"gopro_{camera}")
            ] = sort_files(g)

        if not cae_inventory.empty:
            c = cae_inventory[
                cae_inventory["participant"].eq(participant)
            ].copy()

            # CAE rows without duration cannot provide mechanical offset bounds.
            c = c[
                c["duration_sec_num"].notna()
                & (c["duration_sec_num"] > 0)
            ]

            source_inventory_lookup[
                (participant, "cae_hand")
            ] = sort_files(c)
        else:
            source_inventory_lookup[
                (participant, "cae_hand")
            ] = []

    plan_rows = []

    for participant in blocks["participant"].unique():
        for source in ("gopro_left", "gopro_right", "cae_hand"):
            files = source_inventory_lookup.get(
                (participant, source), []
            )

            plan_rows.extend(
                source_rows_for_participant(
                    participant=participant,
                    blocks=blocks,
                    source=source,
                    files=files,
                    max_group_files=args.max_group_files,
                    fit_tolerance_sec=args.fit_tolerance_sec,
                )
            )

    timing_plan = pd.DataFrame(plan_rows)

    if not timing_plan.empty:
        timing_plan["_p"] = timing_plan["participant"].map(participant_number)
        source_order = {
            "gopro_left": 0,
            "gopro_right": 1,
            "cae_hand": 2,
        }
        timing_plan["_s"] = timing_plan["source"].map(source_order)
        timing_plan = timing_plan.sort_values(
            ["_p", "block_index", "_s"]
        ).drop(columns=["_p", "_s"]).reset_index(drop=True)

    bootstrap = bootstrap_rows(
        attempts,
        timing_plan,
        source_inventory_lookup,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timing_path = out_dir / "timing_block_plan.csv"
    bootstrap_path = out_dir / "attempt_windows_bootstrap.csv"

    timing_plan.to_csv(timing_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)

    print("=== TIMING BLOCK BOOTSTRAP COMPLETE ===")
    print(f"Excluded participants: {', '.join(sorted(excluded, key=participant_number))}")
    print(f"Included participants: {blocks['participant'].nunique()}")
    print(f"Timing blocks: {len(blocks)}")
    print(f"Attempt rows: {len(attempts)}")
    print()
    print(f"Wrote: {timing_path}")
    print(f"Wrote: {bootstrap_path}")
    print()

    print("Timing-block plan by source/status:")
    if len(timing_plan):
        print(
            timing_plan.groupby(
                ["source", "recommended_group_status"]
            ).size().to_string()
        )

    print()
    print("Bootstrap rows by source/status:")
    if len(bootstrap):
        print(
            bootstrap.groupby(
                ["source", "bootstrap_status"]
            ).size().to_string()
        )

    print()
    print("IMPORTANT:")
    print(
        "These are structural/bootstrap ranges only. offset_mid_sec and "
        "bootstrap local times are review starting points, not validated "
        "experimental timestamps."
    )
    print(
        "Do not feed bootstrap windows directly into final inferential "
        "analysis until they have been visually validated/refined."
    )


if __name__ == "__main__":
    main()
