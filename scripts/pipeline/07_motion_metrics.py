from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import angle_deg, ensure_dir, euclid, read_table


def load_reconstructed(video_folder: Path) -> pd.DataFrame:
    candidates = [
        video_folder / "reconstructed.parquet",
        video_folder / "reconstructed.csv.gz",
        video_folder / "reconstructed.csv",
    ]
    for path in candidates:
        if path.exists():
            return read_table(path)
    return pd.DataFrame()


def trajectory(df: pd.DataFrame, landmark_index: int, use_world: bool) -> pd.DataFrame:
    x, y, z = ("world_x", "world_y", "world_z") if use_world else ("x", "y", "z")
    out = df[df["landmark_index"].eq(landmark_index)][
        ["frame_index", "timestamp_sec", x, y, z]
    ].copy()
    out.columns = ["frame_index", "timestamp_sec", "x", "y", "z"]
    for c in ["frame_index", "timestamp_sec", "x", "y", "z"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna().sort_values("timestamp_sec")


def compute_metrics(track: pd.DataFrame) -> dict:
    if len(track) < 3:
        return {}

    xyz = track[["x", "y", "z"]].to_numpy(float)
    t = track["timestamp_sec"].to_numpy(float)
    dt = np.diff(t)
    dxyz = np.diff(xyz, axis=0)
    step = np.linalg.norm(dxyz, axis=1)

    valid = dt > 0
    dt = dt[valid]
    step = step[valid]
    if len(dt) == 0:
        return {}

    velocity = step / dt
    if len(velocity) >= 2:
        dv = np.diff(velocity)
        dtv = dt[1:]
        acceleration = dv / np.where(dtv > 0, dtv, np.nan)
    else:
        acceleration = np.array([])

    if len(acceleration) >= 2:
        da = np.diff(acceleration)
        dta = dt[2:]
        jerk = da / np.where(dta > 0, dta, np.nan)
    else:
        jerk = np.array([])

    displacement = float(np.linalg.norm(xyz[-1] - xyz[0]))
    path_length = float(step.sum())
    efficiency = displacement / path_length if path_length > 0 else np.nan

    return {
        "duration_sec": float(t[-1] - t[0]),
        "straight_line_displacement_3d": displacement,
        "path_length_3d": path_length,
        "path_efficiency_3d": efficiency,
        "mean_velocity": float(np.nanmean(velocity)),
        "median_velocity": float(np.nanmedian(velocity)),
        "peak_velocity": float(np.nanmax(velocity)),
        "mean_acceleration": float(np.nanmean(np.abs(acceleration))) if len(acceleration) else np.nan,
        "peak_acceleration": float(np.nanmax(np.abs(acceleration))) if len(acceleration) else np.nan,
        "mean_abs_jerk": float(np.nanmean(np.abs(jerk))) if len(jerk) else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute attempt-level motion metrics.")
    parser.add_argument("--source-mapping", required=True)
    parser.add_argument("--reconstructed-root", required=True)
    parser.add_argument("--source", required=True, choices=["cae_hand", "gopro_left", "gopro_right"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefer-world", action="store_true")
    args = parser.parse_args()

    mapping = pd.read_csv(args.source_mapping, dtype=str).fillna("")
    mapping = mapping[mapping["source"].eq(args.source)].copy()
    reconstructed_root = Path(args.reconstructed_root)

    rows = []
    for _, m in mapping.iterrows():
        video_id = str(m["selected_video_id"]).strip()
        start = pd.to_numeric(pd.Series([m["local_start_sec"]]), errors="coerce").iloc[0]
        end = pd.to_numeric(pd.Series([m["local_end_sec"]]), errors="coerce").iloc[0]
        if not video_id or pd.isna(start) or pd.isna(end) or end <= start:
            continue

        df = load_reconstructed(reconstructed_root / video_id)
        if df.empty:
            continue

        for col in ["frame_index", "timestamp_sec", "landmark_index"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        hand = m["hand"]
        if "anatomical_hand" in df.columns:
            df = df[df["anatomical_hand"].eq(hand)]

        df = df[(df["timestamp_sec"] >= float(start)) & (df["timestamp_sec"] <= float(end))]
        if df.empty:
            continue

        use_world = args.prefer_world and df[["world_x", "world_y", "world_z"]].apply(
            pd.to_numeric, errors="coerce"
        ).notna().all(axis=1).mean() >= 0.90

        wrist = trajectory(df, 0, use_world)
        index_tip = trajectory(df, 8, use_world)
        thumb_tip = trajectory(df, 4, use_world)

        metrics = compute_metrics(wrist)
        if not metrics:
            continue

        index_metrics = compute_metrics(index_tip)
        thumb_metrics = compute_metrics(thumb_tip)

        row = {
            "attempt_id": m["attempt_id"],
            "participant": m["participant"],
            "round": m["round"],
            "device": m["device"],
            "hand": hand,
            "motion_source": args.source,
            "video_id": video_id,
            "local_start_sec": float(start),
            "local_end_sec": float(end),
            "coordinate_space": "world_xyz" if use_world else "normalized_xyz",
            "tracking_frames": int(wrist["frame_index"].nunique()),
            **metrics,
            "index_tip_displacement_3d": index_metrics.get("straight_line_displacement_3d", np.nan),
            "thumb_tip_displacement_3d": thumb_metrics.get("straight_line_displacement_3d", np.nan),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    dest = Path(args.output)
    ensure_dir(dest.parent)
    out.to_csv(dest, index=False)
    print(f"Wrote: {dest}")
    print(f"Rows: {len(out)}")


if __name__ == "__main__":
    main()
