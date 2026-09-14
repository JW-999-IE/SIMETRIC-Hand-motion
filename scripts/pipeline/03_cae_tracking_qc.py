from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import ensure_dir, read_part_files


def longest_true_run(values: np.ndarray) -> int:
    best = cur = 0
    for v in values:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def longest_false_run(values: np.ndarray) -> int:
    return longest_true_run(~values)


def main():
    parser = argparse.ArgumentParser(description="QC CAE-HAND extraction.")
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--landmarks-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--warn-any-hand-fraction", type=float, default=0.50)
    parser.add_argument("--fail-any-hand-fraction", type=float, default=0.10)
    parser.add_argument("--warn-world-fraction", type=float, default=0.90)
    args = parser.parse_args()

    inventory = pd.read_csv(args.inventory, dtype=str).fillna("")
    inventory = inventory[inventory["camera_source"].eq("cae_hand")].copy()
    landmarks_root = Path(args.landmarks_root)
    out_root = ensure_dir(Path(args.output_root) / "validation")

    rows = []

    for _, video in inventory.iterrows():
        video_id = str(video["video_id"])
        participant = str(video["participant"])
        fps = pd.to_numeric(pd.Series([video.get("fps", "")]), errors="coerce").iloc[0]
        total_frames = pd.to_numeric(pd.Series([video.get("frame_count", "")]), errors="coerce").iloc[0]

        df = read_part_files(landmarks_root / video_id)
        if df.empty:
            rows.append({
                "participant": participant,
                "video_id": video_id,
                "file_name": video["file_name"],
                "qc_status": "FAIL",
                "review_note": "No extracted landmark rows",
            })
            continue

        df["frame_index"] = pd.to_numeric(df["frame_index"], errors="coerce").astype("Int64")
        frames_detected = np.sort(df["frame_index"].dropna().astype(int).unique())
        observed_max = int(frames_detected.max()) if len(frames_detected) else -1

        if pd.isna(total_frames) or int(total_frames) <= 0:
            total_frames = observed_max + 1
        total_frames = int(total_frames)

        per_frame_hands = (
            df[["frame_index", "hand_index"]]
            .drop_duplicates()
            .groupby("frame_index")
            .size()
        )

        presence = np.zeros(total_frames, dtype=bool)
        valid_idx = frames_detected[(frames_detected >= 0) & (frames_detected < total_frames)]
        presence[valid_idx] = True

        two_hand_frames = int((per_frame_hands >= 2).sum())
        any_hand_frames = int(len(frames_detected))
        world_valid = df[["world_x", "world_y", "world_z"]].apply(
            pd.to_numeric, errors="coerce"
        ).notna().all(axis=1).mean()

        raw = df[["frame_index", "hand_index", "raw_handedness", "raw_handedness_score"]].drop_duplicates(
            ["frame_index", "hand_index"]
        )
        score = pd.to_numeric(raw["raw_handedness_score"], errors="coerce")
        ambiguity_fraction = float((score < 0.60).mean()) if len(score) else 1.0

        any_fraction = any_hand_frames / total_frames if total_frames else 0.0
        two_fraction = two_hand_frames / total_frames if total_frames else 0.0
        longest_track = longest_true_run(presence) / fps if fps and fps > 0 else float("nan")
        longest_gap = longest_false_run(presence) / fps if fps and fps > 0 else float("nan")

        if any_fraction < args.fail_any_hand_fraction:
            status = "FAIL"
            note = "Very low hand detection coverage"
        elif any_fraction < args.warn_any_hand_fraction or world_valid < args.warn_world_fraction:
            status = "WARN"
            note = "Tracking coverage and/or world XYZ coverage needs review"
        else:
            status = "PASS"
            note = ""

        rows.append({
            "participant": participant,
            "video_id": video_id,
            "file_name": video["file_name"],
            "duration_sec": video.get("duration_sec", ""),
            "frames_total": total_frames,
            "frames_any_hand": any_hand_frames,
            "frames_two_hands": two_hand_frames,
            "any_hand_fraction": any_fraction,
            "two_hand_fraction": two_fraction,
            "world_xyz_fraction": float(world_valid),
            "minimum_continuous_track_sec": float(longest_track),
            "maximum_tracking_gap_sec": float(longest_gap),
            "handedness_ambiguity_fraction": ambiguity_fraction,
            "qc_status": status,
            "review_note": note,
        })

    qc = pd.DataFrame(rows)
    qc_path = out_root / "cae_tracking_qc.csv"
    qc.to_csv(qc_path, index=False)

    summary = (
        qc.groupby("participant", dropna=False)
        .agg(
            videos=("video_id", "count"),
            pass_videos=("qc_status", lambda s: int((s == "PASS").sum())),
            warn_videos=("qc_status", lambda s: int((s == "WARN").sum())),
            fail_videos=("qc_status", lambda s: int((s == "FAIL").sum())),
            max_any_hand_fraction=("any_hand_fraction", "max"),
            max_two_hand_fraction=("two_hand_fraction", "max"),
        )
        .reset_index()
    )
    summary_path = out_root / "cae_tracking_qc_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Wrote: {qc_path}")
    print(f"Wrote: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
