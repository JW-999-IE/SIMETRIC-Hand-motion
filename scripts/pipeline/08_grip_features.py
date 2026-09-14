from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import angle_deg, ensure_dir, euclid, read_table


def load_reconstructed(video_folder: Path) -> pd.DataFrame:
    for name in ("reconstructed.parquet", "reconstructed.csv.gz", "reconstructed.csv"):
        path = video_folder / name
        if path.exists():
            return read_table(path)
    return pd.DataFrame()


def frame_points(group: pd.DataFrame, use_world: bool):
    cols = ("world_x", "world_y", "world_z") if use_world else ("x", "y", "z")
    points = {}
    for _, row in group.iterrows():
        idx = int(row["landmark_index"])
        vec = np.array([row[cols[0]], row[cols[1]], row[cols[2]]], dtype=float)
        if np.isfinite(vec).all():
            points[idx] = vec
    return points


def classify_candidate(features: dict) -> tuple[str, float]:
    # Deliberately simple, exploratory heuristic. Must remain unvalidated.
    pinch = features.get("thumb_index_norm", np.nan)
    aperture = features.get("hand_aperture_norm", np.nan)
    flex = features.get("mean_finger_flexion_deg", np.nan)

    if np.isfinite(pinch) and pinch < 0.35:
        return "pinch_candidate", 0.65
    if np.isfinite(aperture) and aperture > 1.6 and np.isfinite(flex) and flex > 140:
        return "open_hand_candidate", 0.55
    if np.isfinite(aperture) and aperture < 1.0 and np.isfinite(flex) and flex < 120:
        return "closed_grip_candidate", 0.55
    return "other_candidate", 0.40


def main():
    parser = argparse.ArgumentParser(description="Extract exploratory grip features.")
    parser.add_argument("--source-mapping", required=True)
    parser.add_argument("--reconstructed-root", required=True)
    parser.add_argument("--source", required=True, choices=["cae_hand", "gopro_left", "gopro_right"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefer-world", action="store_true")
    args = parser.parse_args()

    mapping = pd.read_csv(args.source_mapping, dtype=str).fillna("")
    mapping = mapping[mapping["source"].eq(args.source)].copy()
    root = Path(args.reconstructed_root)
    rows = []

    for _, m in mapping.iterrows():
        video_id = str(m["selected_video_id"]).strip()
        start = pd.to_numeric(pd.Series([m["local_start_sec"]]), errors="coerce").iloc[0]
        end = pd.to_numeric(pd.Series([m["local_end_sec"]]), errors="coerce").iloc[0]
        if not video_id or pd.isna(start) or pd.isna(end) or end <= start:
            continue

        df = load_reconstructed(root / video_id)
        if df.empty:
            continue

        for c in ["frame_index", "timestamp_sec", "landmark_index", "x", "y", "z",
                  "world_x", "world_y", "world_z"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if "anatomical_hand" in df.columns:
            df = df[df["anatomical_hand"].eq(m["hand"])]

        df = df[(df["timestamp_sec"] >= float(start)) & (df["timestamp_sec"] <= float(end))]
        if df.empty:
            continue

        use_world = args.prefer_world and df[["world_x", "world_y", "world_z"]].notna().all(axis=1).mean() >= 0.90
        frame_features = []

        for frame_index, group in df.groupby("frame_index"):
            pts = frame_points(group, use_world)
            needed = {0, 4, 8, 12, 16, 20, 5, 9, 13, 17, 6, 10, 14, 18}
            if not needed.issubset(pts):
                continue

            palm_width = euclid(pts[5], pts[17])
            if palm_width <= 0:
                continue

            finger_angles = []
            for a, b, c in [(5,6,8), (9,10,12), (13,14,16), (17,18,20)]:
                finger_angles.append(angle_deg(pts[a], pts[b], pts[c]))

            frame_features.append({
                "thumb_index_norm": euclid(pts[4], pts[8]) / palm_width,
                "thumb_middle_norm": euclid(pts[4], pts[12]) / palm_width,
                "thumb_ring_norm": euclid(pts[4], pts[16]) / palm_width,
                "thumb_little_norm": euclid(pts[4], pts[20]) / palm_width,
                "index_middle_spread_norm": euclid(pts[8], pts[12]) / palm_width,
                "hand_aperture_norm": euclid(pts[4], pts[20]) / palm_width,
                "mean_finger_flexion_deg": float(np.nanmean(finger_angles)),
            })

        if not frame_features:
            continue

        ff = pd.DataFrame(frame_features)
        summary = {col: float(ff[col].median()) for col in ff.columns}
        label, confidence = classify_candidate(summary)

        rows.append({
            "attempt_id": m["attempt_id"],
            "participant": m["participant"],
            "round": m["round"],
            "device": m["device"],
            "hand": m["hand"],
            "grip_source": args.source,
            "video_id": video_id,
            "coordinate_space": "world_xyz" if use_world else "normalized_xyz",
            "feature_frames": len(ff),
            **summary,
            "grip_candidate": label,
            "grip_candidate_confidence": confidence,
            "grip_validated": False,
        })

    out = pd.DataFrame(rows)
    dest = Path(args.output)
    ensure_dir(dest.parent)
    out.to_csv(dest, index=False)
    print(f"Wrote: {dest}")
    print(f"Rows: {len(out)}")


if __name__ == "__main__":
    main()
