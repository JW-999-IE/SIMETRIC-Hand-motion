from __future__ import annotations

import argparse
from pathlib import Path
import math

import numpy as np
import pandas as pd

from common import ensure_dir, read_part_files, write_table


def adjusted_label(raw: str, invert: bool) -> str:
    raw = str(raw).strip().lower()
    if raw not in {"left", "right"}:
        return ""
    if invert:
        return "right" if raw == "left" else "left"
    return raw


def dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def choose_assignments(candidates, previous, invert, label_weight=0.30, distance_weight=1.0):
    """
    Assign detected hands to anatomical left/right using raw handedness evidence
    plus wrist continuity. This does not prove anatomical semantics; use
    --invert-handedness after calibration if needed.
    """
    if not candidates:
        return {}

    labels = ["left", "right"]

    def cost(candidate, target):
        label = adjusted_label(candidate["raw_handedness"], invert)
        score = float(candidate["raw_handedness_score"]) if np.isfinite(candidate["raw_handedness_score"]) else 0.5
        label_cost = 0.0 if label == target else (1.0 * score if label else 0.5)
        prev = previous.get(target)
        d = 0.0 if prev is None else dist(candidate["wrist"], prev)
        return label_weight * label_cost + distance_weight * d

    if len(candidates) == 1:
        c = candidates[0]
        label = adjusted_label(c["raw_handedness"], invert)
        if label:
            return {c["hand_index"]: label}
        # Fall back to nearest previous identity.
        options = [(cost(c, target), target) for target in labels]
        options.sort()
        return {c["hand_index"]: options[0][1]}

    # Use best two candidates by handedness score if somehow >2.
    candidates = sorted(
        candidates,
        key=lambda x: (-(x["raw_handedness_score"] if np.isfinite(x["raw_handedness_score"]) else 0.0))
    )[:2]
    c0, c1 = candidates

    cost_lr = cost(c0, "left") + cost(c1, "right")
    cost_rl = cost(c0, "right") + cost(c1, "left")

    if cost_lr <= cost_rl:
        return {c0["hand_index"]: "left", c1["hand_index"]: "right"}
    return {c0["hand_index"]: "right", c1["hand_index"]: "left"}


def main():
    parser = argparse.ArgumentParser(description="Reconstruct CAE anatomical hand identities.")
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--landmarks-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--participants", nargs="*")
    parser.add_argument("--invert-handedness", action="store_true",
                        help="Use only after a visual calibration shows MediaPipe labels are mirrored.")
    args = parser.parse_args()

    inventory = pd.read_csv(args.inventory, dtype=str).fillna("")
    inventory = inventory[inventory["camera_source"].eq("cae_hand")].copy()

    if args.participants:
        wanted = {x.upper() for x in args.participants}
        inventory = inventory[inventory["participant"].str.upper().isin(wanted)]

    landmarks_root = Path(args.landmarks_root)
    reconstructed_root = ensure_dir(Path(args.output_root) / "analysis" / "reconstructed")
    qc_rows = []

    for _, video in inventory.iterrows():
        video_id = str(video["video_id"])
        participant = str(video["participant"])
        df = read_part_files(landmarks_root / video_id)
        if df.empty:
            continue

        for col in ["frame_index", "hand_index", "landmark_index", "raw_handedness_score",
                    "x", "y", "z", "world_x", "world_y", "world_z"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        wrists = df[df["landmark_index"].eq(0)].copy()
        previous = {}
        assignments = {}
        ambiguity = 0
        swap_like = 0
        prior_frame_assignment = {}

        for frame_index, group in wrists.groupby("frame_index", sort=True):
            candidates = []
            for _, row in group.iterrows():
                candidates.append({
                    "hand_index": int(row["hand_index"]),
                    "raw_handedness": row.get("raw_handedness", ""),
                    "raw_handedness_score": float(row.get("raw_handedness_score", np.nan)),
                    "wrist": np.array([row["x"], row["y"], row["z"]], dtype=float),
                })

            assigned = choose_assignments(candidates, previous, args.invert_handedness)
            assignments[int(frame_index)] = assigned

            for c in candidates:
                anatomical = assigned.get(c["hand_index"])
                if not anatomical:
                    continue
                score = c["raw_handedness_score"]
                if not np.isfinite(score) or score < 0.60:
                    ambiguity += 1
                if prior_frame_assignment.get(c["hand_index"]) not in {None, anatomical}:
                    swap_like += 1
                prior_frame_assignment[c["hand_index"]] = anatomical
                previous[anatomical] = c["wrist"]

        def label_row(row):
            return assignments.get(int(row["frame_index"]), {}).get(int(row["hand_index"]), "unresolved")

        df["anatomical_hand"] = df.apply(label_row, axis=1)
        df["handedness_mode"] = "inverted" if args.invert_handedness else "as_reported"

        out_folder = ensure_dir(reconstructed_root / video_id)
        written = write_table(df, out_folder / "reconstructed.parquet")

        hand_frames = (
            df[["frame_index", "anatomical_hand"]]
            .drop_duplicates()
        )
        total_detected_frames = hand_frames["frame_index"].nunique()
        left_frames = hand_frames[hand_frames["anatomical_hand"].eq("left")]["frame_index"].nunique()
        right_frames = hand_frames[hand_frames["anatomical_hand"].eq("right")]["frame_index"].nunique()
        detected_hands = len(wrists)

        qc_rows.append({
            "participant": participant,
            "video_id": video_id,
            "file_name": video["file_name"],
            "output_file": str(written),
            "left_detected_frames": left_frames,
            "right_detected_frames": right_frames,
            "detected_frames": total_detected_frames,
            "left_coverage_of_detected_frames": left_frames / total_detected_frames if total_detected_frames else 0.0,
            "right_coverage_of_detected_frames": right_frames / total_detected_frames if total_detected_frames else 0.0,
            "handedness_ambiguity_fraction": ambiguity / detected_hands if detected_hands else 1.0,
            "swap_like_events": swap_like,
            "handedness_mode": "inverted" if args.invert_handedness else "as_reported",
            "qc_status": "WARN" if ambiguity / max(detected_hands, 1) > 0.25 else "PASS",
        })

    qc = pd.DataFrame(qc_rows)
    qc_path = ensure_dir(Path(args.output_root) / "validation") / "cae_identity_qc.csv"
    qc.to_csv(qc_path, index=False)
    print(f"Wrote: {qc_path}")
    if not qc.empty:
        print(qc.to_string(index=False))


if __name__ == "__main__":
    main()
