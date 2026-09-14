from __future__ import annotations

import argparse
from pathlib import Path
import math
import time

import cv2
import mediapipe as mp
import pandas as pd

from common import (
    ensure_dir,
    load_json,
    parse_participants,
    save_json,
    write_table,
)


LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]


def main():
    parser = argparse.ArgumentParser(description="Pilot CAE-HAND MediaPipe extraction.")
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--participants", nargs="+", required=True)
    parser.add_argument("--chunk-frames", type=int, default=5000)
    parser.add_argument("--max-minutes", type=float, default=0.0,
                        help="0 = full video. Use a positive value for a short pilot.")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    args = parser.parse_args()

    inventory = pd.read_csv(args.inventory, dtype=str).fillna("")
    participants = parse_participants(args.participants)
    inventory = inventory[
        inventory["camera_source"].eq("cae_hand")
        & inventory["participant"].str.upper().isin(participants)
    ].copy()

    if inventory.empty:
        raise SystemExit("No matching CAE-HAND videos in inventory.")

    out_root = Path(args.output_root)
    landmarks_root = ensure_dir(out_root / "landmarks")
    validation_root = ensure_dir(out_root / "validation")

    mp_hands = mp.solutions.hands
    results = []

    for _, video in inventory.iterrows():
        video_id = str(video["video_id"])
        participant = str(video["participant"])
        path = Path(video["path"])
        video_out = ensure_dir(landmarks_root / video_id)
        checkpoint_path = video_out / "checkpoint.json"
        checkpoint = load_json(checkpoint_path, default={}) or {}
        start_frame = int(checkpoint.get("next_frame", 0))

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            results.append({
                "participant": participant,
                "video_id": video_id,
                "file_name": path.name,
                "status": "open_failed",
                "frames_processed": 0,
                "total_frames": "",
            })
            continue

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if args.max_minutes > 0 and fps > 0:
            total_frames = min(
                total_frames,
                int(args.max_minutes * 60.0 * fps),
            )

        if start_frame >= total_frames and total_frames > 0:
            print(f"[{video_id}] already complete")
            cap.release()
            results.append({
                "participant": participant,
                "video_id": video_id,
                "file_name": path.name,
                "status": "skipped_complete",
                "frames_processed": total_frames,
                "total_frames": total_frames,
            })
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"[{video_id}] {participant} {path.name} starting frame {start_frame}/{total_frames}")

        chunk_rows = []
        part_index = int(checkpoint.get("part_index", 0))
        frames_processed = start_frame
        started = time.time()

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        ) as hands:
            frame_index = start_frame

            while frame_index < total_frames:
                ok, frame = cap.read()
                if not ok:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                timestamp_sec = (frame_index / fps) if fps > 0 else float("nan")

                hand_landmarks = result.multi_hand_landmarks or []
                world_landmarks = result.multi_hand_world_landmarks or []
                handedness = result.multi_handedness or []

                for hand_index, lm_list in enumerate(hand_landmarks):
                    raw_label = ""
                    raw_score = float("nan")
                    if hand_index < len(handedness) and handedness[hand_index].classification:
                        c = handedness[hand_index].classification[0]
                        raw_label = c.label
                        raw_score = float(c.score)

                    world = world_landmarks[hand_index] if hand_index < len(world_landmarks) else None

                    for landmark_index, lm in enumerate(lm_list.landmark):
                        w = world.landmark[landmark_index] if world is not None else None
                        chunk_rows.append({
                            "participant": participant,
                            "video_id": video_id,
                            "source": "cae_hand",
                            "frame_index": frame_index,
                            "timestamp_sec": timestamp_sec,
                            "hand_index": hand_index,
                            "raw_handedness": raw_label,
                            "raw_handedness_score": raw_score,
                            "landmark_index": landmark_index,
                            "landmark_name": LANDMARK_NAMES[landmark_index],
                            "x": float(lm.x),
                            "y": float(lm.y),
                            "z": float(lm.z),
                            "visibility": float(getattr(lm, "visibility", float("nan"))),
                            "presence": float(getattr(lm, "presence", float("nan"))),
                            "world_x": float(w.x) if w is not None else float("nan"),
                            "world_y": float(w.y) if w is not None else float("nan"),
                            "world_z": float(w.z) if w is not None else float("nan"),
                        })

                frame_index += 1
                frames_processed = frame_index

                if frame_index % args.chunk_frames == 0 or frame_index >= total_frames:
                    part = pd.DataFrame(chunk_rows)
                    if not part.empty:
                        written = write_table(part, video_out / f"part-{part_index:05d}.parquet")
                        print(f"[{video_id}] wrote {written.name} through frame {frame_index - 1}")
                    else:
                        # still persist an empty-frame checkpoint
                        print(f"[{video_id}] no hands in chunk ending frame {frame_index - 1}")
                    chunk_rows = []
                    part_index += 1
                    save_json(checkpoint_path, {
                        "next_frame": frame_index,
                        "part_index": part_index,
                        "fps": fps,
                        "total_frames": total_frames,
                        "source_file": str(path),
                    })

        cap.release()
        elapsed = max(time.time() - started, 1e-6)
        status = "complete" if frames_processed >= total_frames else "partial"
        results.append({
            "participant": participant,
            "video_id": video_id,
            "file_name": path.name,
            "status": status,
            "frames_processed": frames_processed,
            "total_frames": total_frames,
            "processing_fps": (frames_processed - start_frame) / elapsed,
        })

    result_df = pd.DataFrame(results)
    result_path = validation_root / "extraction_result.csv"
    result_df.to_csv(result_path, index=False)
    print(f"Wrote: {result_path}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
