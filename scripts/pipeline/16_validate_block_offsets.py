from __future__ import annotations

import argparse
import math
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd


FINAL_STATUSES = {
    "validated",
    "validated_fragment",
    "not_usable",
    "no_source",
}


def participant_number(value: object) -> int:
    m = re.search(r"(\d+)", str(value))
    return int(m.group(1)) if m else 999999


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str).fillna("")


def as_float(value, default=float("nan")) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def as_int(value, default=999999) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def backup_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.stem}.backup-{stamp}{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def save_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def split_semicolon(value: object) -> list[str]:
    return [x for x in str(value).split(";") if x != ""]


def anchor_video_path(row: pd.Series) -> str:
    anchor_file = str(row.get("anchor_video_file", ""))
    files = split_semicolon(row.get("recommended_video_files", ""))
    paths = split_semicolon(row.get("recommended_video_paths", ""))

    if anchor_file and len(files) == len(paths):
        for file_name, path in zip(files, paths):
            if file_name == anchor_file:
                return path

    if len(paths) == 1:
        return paths[0]

    return ""


def selected_video_path(row: pd.Series, file_index: int = 0) -> tuple[str, str]:
    files = split_semicolon(row.get("recommended_video_files", ""))
    paths = split_semicolon(row.get("recommended_video_paths", ""))

    if len(files) == len(paths) and paths:
        file_index %= len(paths)
        return files[file_index], paths[file_index]

    path = anchor_video_path(row)
    return str(row.get("anchor_video_file", "")), path


def open_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frames / fps if fps > 0 else 0.0

    return cap, fps, frames, duration


def seek(cap, sec: float, duration: float) -> float:
    sec = max(0.0, min(float(sec), max(duration - 0.001, 0.0)))
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
    return sec


def draw_overlay(
    frame,
    row: pd.Series,
    file_name: str,
    current_sec: float,
    duration: float,
    marked_start: float | None,
    paused: bool,
):
    queue_id = str(row.get("queue_id", ""))
    participant = str(row.get("participant", ""))
    block_id = str(row.get("block_id", ""))
    source = str(row.get("source", ""))
    anchor_attempt = str(row.get("anchor_attempt_id", ""))
    ref_start = str(row.get("anchor_reference_start_sec", ""))
    mid = str(row.get("anchor_bootstrap_local_sec", ""))
    uncertainty = str(row.get("offset_uncertainty_sec", ""))

    lines = [
        f"{queue_id} | {participant} | {block_id} | {source}",
        f"anchor attempt: {anchor_attempt}",
        f"video: {file_name}",
        f"time: {current_sec:.3f}s / {duration:.3f}s | {'PAUSED' if paused else 'PLAYING'}",
        f"bootstrap anchor local: {mid}s | offset uncertainty: {uncertainty}s",
        f"reference anchor start: {ref_start}s",
        f"MARKED actual anchor start: {'--' if marked_start is None else f'{marked_start:.3f}s'}",
        "SPACE play/pause | j/l -/+5s | J/L -/+30s | ,/. frame",
        "s = mark actual anchor start | v = validate/save offset",
        "n = next video in group | x = uncertain | u = not usable | q = save+quit",
    ]

    y = 28
    for line in lines:
        cv2.putText(
            frame, line, (15, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            frame, line, (15, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 0), 1, cv2.LINE_AA
        )
        y += 25

    return frame


def initial_seek_for_row(row: pd.Series) -> float:
    for key in (
        "review_seek_start_sec",
        "anchor_bootstrap_local_sec",
    ):
        value = as_float(row.get(key, ""))
        if math.isfinite(value):
            return max(value, 0.0)
    return 0.0


def validate_offset(
    row: pd.Series,
    actual_anchor_local_sec: float,
) -> tuple[float, float, float]:
    reference_anchor = as_float(
        row.get("anchor_reference_start_sec", "")
    )
    if not math.isfinite(reference_anchor):
        raise ValueError(
            "anchor_reference_start_sec is missing; cannot calculate offset."
        )

    validated_offset = reference_anchor - actual_anchor_local_sec

    ref_block_start = as_float(
        row.get("reference_min_start_sec", "")
    )
    ref_block_end = as_float(
        row.get("reference_max_end_sec", "")
    )

    block_start = (
        ref_block_start - validated_offset
        if math.isfinite(ref_block_start)
        else float("nan")
    )
    block_end = (
        ref_block_end - validated_offset
        if math.isfinite(ref_block_end)
        else float("nan")
    )

    return validated_offset, block_start, block_end


def mark_validated(
    df: pd.DataFrame,
    idx: int,
    marked_start: float,
    reviewer: str,
):
    row = df.loc[idx]

    validated_offset, block_start, block_end = validate_offset(
        row,
        marked_start,
    )

    old_min = as_float(row.get("offset_min_sec", ""))
    old_max = as_float(row.get("offset_max_sec", ""))

    if (
        math.isfinite(old_min)
        and math.isfinite(old_max)
        and not (old_min <= validated_offset <= old_max)
    ):
        warning = (
            f"Validated offset {validated_offset:.3f}s lies outside "
            f"bootstrap range [{old_min:.3f}, {old_max:.3f}]s."
        )
    else:
        warning = ""

    df.at[idx, "validation_status"] = "validated"
    df.at[idx, "validated_anchor_local_sec"] = f"{marked_start:.6f}"
    df.at[idx, "validated_offset_sec"] = f"{validated_offset:.6f}"
    df.at[idx, "validated_block_start_local_sec"] = (
        "" if not math.isfinite(block_start) else f"{block_start:.6f}"
    )
    df.at[idx, "validated_block_end_local_sec"] = (
        "" if not math.isfinite(block_end) else f"{block_end:.6f}"
    )
    df.at[idx, "reviewer"] = reviewer
    df.at[idx, "validated_at"] = datetime.now().isoformat(
        timespec="seconds"
    )

    if warning:
        previous = str(df.at[idx, "review_note"])
        df.at[idx, "review_note"] = (
            (previous + " " if previous else "") + warning
        )

    return validated_offset, block_start, block_end, warning


def audit_queue(df: pd.DataFrame) -> None:
    print("=== BLOCK OFFSET VALIDATION STATUS ===")
    print(f"Queue rows: {len(df)}")
    print()

    print("Validation status:")
    print(
        df["validation_status"]
        .replace("", "<blank>")
        .value_counts()
        .to_string()
    )
    print()

    unresolved = ~df["validation_status"].isin(FINAL_STATUSES)
    print(f"Resolved: {(~unresolved).sum()}")
    print(f"Remaining: {unresolved.sum()}")
    print()

    if unresolved.any():
        by_priority = (
            df.loc[unresolved]
            .groupby(["priority", "source", "review_type"])
            .size()
        )
        print("Remaining by priority/source/type:")
        print(by_priority.to_string())


def select_rows(
    df: pd.DataFrame,
    queue_ids: set[str] | None,
    participants: set[str] | None,
    priorities: set[str] | None,
    sources: set[str] | None,
    only_unresolved: bool,
) -> list[int]:
    mask = pd.Series(True, index=df.index)

    if queue_ids:
        mask &= df["queue_id"].isin(queue_ids)

    if participants:
        mask &= df["participant"].str.upper().isin(participants)

    if priorities:
        mask &= df["priority"].astype(str).isin(priorities)

    if sources:
        mask &= df["source"].isin(sources)

    if only_unresolved:
        mask &= ~df["validation_status"].isin(FINAL_STATUSES)

    selected = df[mask].copy()
    selected["_p"] = selected["participant"].map(participant_number)
    selected["_block"] = selected["block_index"].map(as_int)
    selected["_priority"] = selected["priority"].map(as_int)

    selected = selected.sort_values(
        ["_priority", "_p", "_block", "source"]
    )

    return list(selected.index)


def interactive_validate(
    df: pd.DataFrame,
    queue_path: Path,
    indices: list[int],
    reviewer: str,
):
    if not indices:
        print("Nothing selected for validation.")
        return

    print(f"Selected queue rows: {len(indices)}")
    print("The queue CSV is saved after every decision.")
    print()

    window_name = "SIMETRIC block offset validator"

    for pos, idx in enumerate(indices, 1):
        row = df.loc[idx]

        print()
        print(
            f"[{pos}/{len(indices)}] "
            f"{row.get('queue_id','')} "
            f"{row.get('participant','')} "
            f"{row.get('block_id','')} "
            f"{row.get('source','')}"
        )
        print(
            f"Anchor attempt: {row.get('anchor_attempt_id','')}"
        )
        print(
            f"Bootstrap anchor: "
            f"{row.get('anchor_video_file','')} @ "
            f"{row.get('anchor_bootstrap_local_sec','')}s"
        )

        files = split_semicolon(
            row.get("recommended_video_files", "")
        )
        paths = split_semicolon(
            row.get("recommended_video_paths", "")
        )

        anchor_path = anchor_video_path(row)

        if not paths and anchor_path:
            paths = [anchor_path]
            files = [str(row.get("anchor_video_file", ""))]

        if not paths:
            print("No video path in queue row.")
            df.at[idx, "validation_status"] = "uncertain"
            note = str(df.at[idx, "review_note"])
            extra = "Validator could not find a video path."
            df.at[idx, "review_note"] = (
                f"{note} {extra}".strip()
            )
            save_atomic(df, queue_path)
            continue

        if len(files) != len(paths):
            print("Video files/paths count mismatch.")
            df.at[idx, "validation_status"] = "uncertain"
            save_atomic(df, queue_path)
            continue

        anchor_file = str(row.get("anchor_video_file", ""))
        file_index = 0

        if anchor_file in files:
            file_index = files.index(anchor_file)

        marked_start = as_float(
            row.get("validated_anchor_local_sec", "")
        )
        if not math.isfinite(marked_start):
            marked_start = None

        row_done = False
        quit_all = False

        while not row_done:
            file_name = files[file_index]
            path = paths[file_index]

            try:
                cap, fps, frame_count, duration = open_video(path)
            except Exception as exc:
                print(exc)

                if len(paths) > 1:
                    file_index = (file_index + 1) % len(paths)
                    if file_index == 0:
                        df.at[idx, "validation_status"] = "uncertain"
                        save_atomic(df, queue_path)
                        row_done = True
                    continue

                df.at[idx, "validation_status"] = "uncertain"
                save_atomic(df, queue_path)
                break

            if file_name == anchor_file:
                current_sec = initial_seek_for_row(row)
            else:
                current_sec = 0.0

            current_sec = seek(cap, current_sec, duration)
            paused = True
            last_frame = None
            switch_video = False

            while True:
                if not paused or last_frame is None:
                    ok, frame = cap.read()

                    if not ok:
                        paused = True
                        current_sec = seek(
                            cap,
                            max(duration - 0.1, 0.0),
                            duration,
                        )
                        last_frame = None
                        continue

                    last_frame = frame
                    frame_no = (
                        cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
                    )

                    if fps > 0:
                        current_sec = frame_no / fps
                    else:
                        current_sec = (
                            cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        )

                display = draw_overlay(
                    last_frame.copy(),
                    row,
                    file_name,
                    current_sec,
                    duration,
                    marked_start,
                    paused,
                )

                cv2.imshow(window_name, display)
                key = cv2.waitKeyEx(30 if not paused else 0)

                if key == -1:
                    continue

                low = key & 0xFF

                if low == ord(" "):
                    paused = not paused

                elif low == ord("j"):
                    current_sec = seek(
                        cap, current_sec - 5.0, duration
                    )
                    last_frame = None

                elif low == ord("l"):
                    current_sec = seek(
                        cap, current_sec + 5.0, duration
                    )
                    last_frame = None

                elif low == ord("J"):
                    current_sec = seek(
                        cap, current_sec - 30.0, duration
                    )
                    last_frame = None

                elif low == ord("L"):
                    current_sec = seek(
                        cap, current_sec + 30.0, duration
                    )
                    last_frame = None

                elif low == ord(","):
                    paused = True
                    current_sec = seek(
                        cap,
                        current_sec - (
                            1.0 / fps if fps > 0 else 0.033
                        ),
                        duration,
                    )
                    last_frame = None

                elif low == ord("."):
                    paused = True
                    current_sec = seek(
                        cap,
                        current_sec + (
                            1.0 / fps if fps > 0 else 0.033
                        ),
                        duration,
                    )
                    last_frame = None

                elif low == ord("s"):
                    marked_start = current_sec
                    print(
                        f"Marked actual anchor start: "
                        f"{marked_start:.3f}s"
                    )

                elif low == ord("n"):
                    file_index = (
                        file_index + 1
                    ) % len(paths)
                    switch_video = True
                    break

                elif low == ord("v"):
                    if marked_start is None:
                        print(
                            "NOT SAVED: press 's' on the actual "
                            "anchor attempt start first."
                        )
                        continue

                    try:
                        (
                            validated_offset,
                            block_start,
                            block_end,
                            warning,
                        ) = mark_validated(
                            df,
                            idx,
                            marked_start,
                            reviewer,
                        )
                    except ValueError as exc:
                        print(f"NOT SAVED: {exc}")
                        continue

                    save_atomic(df, queue_path)

                    print(
                        f"VALIDATED {row.get('queue_id','')}: "
                        f"offset={validated_offset:.3f}s"
                    )

                    if math.isfinite(block_start):
                        print(
                            f"Block virtual start="
                            f"{block_start:.3f}s"
                        )

                    if math.isfinite(block_end):
                        print(
                            f"Block virtual end="
                            f"{block_end:.3f}s"
                        )

                    if warning:
                        print("WARNING:", warning)

                    row_done = True
                    break

                elif low == ord("x"):
                    df.at[idx, "validation_status"] = "uncertain"
                    df.at[idx, "reviewer"] = reviewer
                    df.at[idx, "validated_at"] = (
                        datetime.now().isoformat(timespec="seconds")
                    )
                    save_atomic(df, queue_path)
                    print(
                        f"UNCERTAIN {row.get('queue_id','')}"
                    )
                    row_done = True
                    break

                elif low == ord("u"):
                    df.at[idx, "validation_status"] = "not_usable"
                    df.at[idx, "reviewer"] = reviewer
                    df.at[idx, "validated_at"] = (
                        datetime.now().isoformat(timespec="seconds")
                    )
                    save_atomic(df, queue_path)
                    print(
                        f"NOT USABLE {row.get('queue_id','')}"
                    )
                    row_done = True
                    break

                elif low == ord("q"):
                    save_atomic(df, queue_path)
                    quit_all = True
                    break

            cap.release()

            if quit_all:
                cv2.destroyAllWindows()
                print("Progress saved.")
                return

            if switch_video:
                continue

        if row_done:
            continue

    cv2.destroyAllWindows()
    save_atomic(df, queue_path)
    print("Selected validation work complete.")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Interactively validate block-level temporal offsets from "
            "block_validation_queue.csv. One visually identified anchor "
            "attempt start yields one validated block offset."
        )
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("audit")
    p.add_argument("--queue", required=True)

    p = sub.add_parser("validate")
    p.add_argument("--queue", required=True)
    p.add_argument("--queue-ids", nargs="*")
    p.add_argument("--participants", nargs="*")
    p.add_argument("--priorities", nargs="*")
    p.add_argument(
        "--sources",
        nargs="*",
        choices=["gopro_left", "gopro_right", "cae_hand"],
    )
    p.add_argument("--reviewer", default="")
    p.add_argument(
        "--all",
        action="store_true",
        help="Include already-resolved queue rows too.",
    )

    args = ap.parse_args()

    queue_path = Path(args.queue)
    df = read_csv(queue_path)

    required = {
        "queue_id",
        "priority",
        "participant",
        "block_id",
        "source",
        "anchor_attempt_id",
        "anchor_reference_start_sec",
        "anchor_video_file",
        "anchor_bootstrap_local_sec",
        "recommended_video_files",
        "recommended_video_paths",
        "reference_min_start_sec",
        "reference_max_end_sec",
        "validation_status",
        "validated_anchor_local_sec",
        "validated_offset_sec",
    }
    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            "block_validation_queue.csv is missing columns: "
            + ", ".join(sorted(missing))
        )

    if "validated_at" not in df.columns:
        df["validated_at"] = ""

    if args.command == "audit":
        audit_queue(df)
        return

    queue_ids = (
        set(args.queue_ids)
        if args.queue_ids
        else None
    )

    participants = (
        {x.upper() for x in args.participants}
        if args.participants
        else None
    )

    priorities = (
        {str(x) for x in args.priorities}
        if args.priorities
        else None
    )

    sources = (
        set(args.sources)
        if args.sources
        else None
    )

    indices = select_rows(
        df,
        queue_ids=queue_ids,
        participants=participants,
        priorities=priorities,
        sources=sources,
        only_unresolved=not args.all,
    )

    if not indices:
        print("Nothing selected.")
        return

    backup = backup_file(queue_path)
    if backup:
        print(f"Backup created: {backup}")

    interactive_validate(
        df,
        queue_path,
        indices,
        reviewer=args.reviewer,
    )


if __name__ == "__main__":
    main()
