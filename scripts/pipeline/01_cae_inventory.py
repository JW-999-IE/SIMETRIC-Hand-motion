from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from common import (
    discover_cae_hand_videos,
    discover_cae_room_videos,
    ensure_dir,
    infer_participant,
    parse_participants,
    stable_video_id,
)


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {
            "probe_status": "failed",
            "fps": "",
            "frame_count": "",
            "width": "",
            "height": "",
            "duration_sec": "",
        }

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    duration = (frames / fps) if fps > 0 else 0.0
    return {
        "probe_status": "ok",
        "fps": fps,
        "frame_count": frames,
        "width": width,
        "height": height,
        "duration_sec": duration,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Inventory CAE videos without touching the existing GoPro inventory."
    )
    parser.add_argument("--cae-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--participants", nargs="*")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Open HAND videos with OpenCV to read fps/frame count. This may hydrate cloud files.",
    )
    parser.add_argument(
        "--include-room-metadata",
        action="store_true",
        help="List ROOM files but never probe/open them.",
    )
    args = parser.parse_args()

    cae_root = Path(args.cae_root)
    out_root = Path(args.output_root)
    participants = parse_participants(args.participants)

    hand = discover_cae_hand_videos(cae_root, participants)
    room = discover_cae_room_videos(cae_root, participants) if args.include_room_metadata else []

    rows = []
    for path in hand:
        row = {
            "video_id": stable_video_id(path),
            "participant": infer_participant(path),
            "camera_source": "cae_hand",
            "file_name": path.name,
            "path": str(path),
            "file_size_bytes": path.stat().st_size,
            "probe_status": "not_probed",
            "fps": "",
            "frame_count": "",
            "width": "",
            "height": "",
            "duration_sec": "",
        }
        if args.probe:
            row.update(probe_video(path))
        rows.append(row)

    for path in room:
        rows.append({
            "video_id": stable_video_id(path),
            "participant": infer_participant(path),
            "camera_source": "cae_room",
            "file_name": path.name,
            "path": str(path),
            "file_size_bytes": path.stat().st_size,
            "probe_status": "not_probed_room",
            "fps": "",
            "frame_count": "",
            "width": "",
            "height": "",
            "duration_sec": "",
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["_p"] = df["participant"].str.extract(r"(\d+)")[0].astype(float)
        df = df.sort_values(["_p", "camera_source", "file_name"]).drop(columns="_p")

    dest = ensure_dir(out_root / "inventory") / "cae_video_inventory.csv"
    df.to_csv(dest, index=False)

    print(f"Wrote: {dest}")
    print(f"Rows: {len(df)}")
    if not df.empty:
        print(df.groupby(["camera_source", "probe_status"]).size().to_string())


if __name__ == "__main__":
    main()
