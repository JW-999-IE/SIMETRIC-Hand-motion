from __future__ import annotations

import argparse
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd


def participant_number(value: object) -> int:
    import re
    m = re.search(r"(\d+)", str(value))
    return int(m.group(1)) if m else 999999


def parse_manual_target(value: str) -> tuple[str, str]:
    try:
        participant, hand = value.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Manual targets must look like P18:left or P21:right"
        ) from exc
    participant = participant.strip().upper()
    hand = hand.strip().lower()
    if hand not in {"left", "right"}:
        raise argparse.ArgumentTypeError("Hand must be left or right")
    return participant, hand


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_landmark_parts(folder: Path) -> pd.DataFrame:
    parquet = sorted(folder.glob("part-*.parquet"))
    csvgz = sorted(folder.glob("part-*.csv.gz"))
    csv = sorted(folder.glob("part-*.csv"))
    paths = parquet or csvgz or csv

    frames = []
    for path in paths:
        if path.suffix.lower() == ".parquet":
            frames.append(pd.read_parquet(path))
        elif path.name.endswith(".csv.gz"):
            frames.append(pd.read_csv(path, compression="gzip"))
        else:
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def read_reconstructed(folder: Path) -> pd.DataFrame:
    candidates = [
        folder / "reconstructed.parquet",
        folder / "reconstructed.csv.gz",
        folder / "reconstructed.csv",
    ]
    for path in candidates:
        if path.exists():
            if path.suffix.lower() == ".parquet":
                return pd.read_parquet(path)
            if path.name.endswith(".csv.gz"):
                return pd.read_csv(path, compression="gzip")
            return pd.read_csv(path)
    return pd.DataFrame()


def longest_run(indices: np.ndarray) -> int:
    if len(indices) == 0:
        return 0
    indices = np.sort(np.unique(indices.astype(int)))
    best = cur = 1
    for a, b in zip(indices[:-1], indices[1:]):
        if b == a + 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def build_recovery_targets(
    gopro_inventory: pd.DataFrame,
    mapping_v2: pd.DataFrame,
    ratio_threshold: float,
    manual_targets: list[tuple[str, str]],
) -> pd.DataFrame:
    expected_participants = sorted(
        mapping_v2["participant"].astype(str).unique(),
        key=participant_number,
    )

    inv = gopro_inventory.copy()
    inv["duration_sec_num"] = pd.to_numeric(inv["duration_sec"], errors="coerce").fillna(0.0)

    # Match the original inventory semantics.
    usable = inv[
        inv["camera"].astype(str).str.lower().isin(["left", "right"])
        & inv["probe_status"].astype(str).eq("ok")
        & ~inv["excluded"].astype(str).str.lower().isin(["true", "1", "yes"])
    ].copy()

    grouped = (
        usable.groupby(["participant", "camera"], dropna=False)
        .agg(
            gopro_video_count=("video_id", "count"),
            gopro_total_duration_sec=("duration_sec_num", "sum"),
        )
        .reset_index()
    )

    lookup = {
        (str(r.participant), str(r.camera).lower()): (
            int(r.gopro_video_count),
            float(r.gopro_total_duration_sec),
        )
        for r in grouped.itertuples(index=False)
    }

    manual_lookup = {}
    for participant, hand in manual_targets:
        manual_lookup.setdefault((participant, hand), []).append("manual_override")

    rows = []
    for participant in expected_participants:
        counts = {}
        durations = {}
        for hand in ("left", "right"):
            count, duration = lookup.get((participant, hand), (0, 0.0))
            counts[hand] = count
            durations[hand] = duration

        for hand in ("left", "right"):
            other = "right" if hand == "left" else "left"
            reasons = []

            if counts[hand] == 0:
                reasons.append("no_usable_gopro_video")

            # Detect major left/right recording asymmetry without using workbook timestamps.
            if (
                counts[hand] > 0
                and durations[other] > 0
                and durations[hand] < ratio_threshold * durations[other]
            ):
                reasons.append(
                    f"duration_asymmetry_{durations[hand]:.1f}s_vs_{durations[other]:.1f}s"
                )

            reasons.extend(manual_lookup.get((participant.upper(), hand), []))

            if reasons:
                rows.append({
                    "participant": participant,
                    "missing_hand": hand,
                    "gopro_video_count": counts[hand],
                    "gopro_total_duration_sec": round(durations[hand], 3),
                    "counterpart_gopro_video_count": counts[other],
                    "counterpart_gopro_total_duration_sec": round(durations[other], 3),
                    "target_reason": ";".join(dict.fromkeys(reasons)),
                })

    targets = pd.DataFrame(rows)
    if not targets.empty:
        targets["_p"] = targets["participant"].map(participant_number)
        targets = targets.sort_values(["_p", "missing_hand"]).drop(columns="_p")
    return targets


def run_command(cmd: list[str]) -> None:
    print()
    print(">>>", subprocess.list2cmdline(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(
            f"Command failed with exit code {result.returncode}: "
            f"{subprocess.list2cmdline(cmd)}"
        )


def build_recovery_qc(
    targets: pd.DataFrame,
    cae_inventory: pd.DataFrame,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    landmark_root = output_root / "landmarks"
    reconstructed_root = output_root / "analysis" / "reconstructed"

    video_rows = []

    hand_videos = cae_inventory[
        cae_inventory["camera_source"].astype(str).eq("cae_hand")
    ].copy()

    for _, video in hand_videos.iterrows():
        participant = str(video["participant"])
        video_id = str(video["video_id"])
        file_name = str(video["file_name"])

        folder = landmark_root / video_id
        checkpoint = read_json(folder / "checkpoint.json", {}) or {}

        processed_frames = int(checkpoint.get("next_frame", 0) or 0)
        total_frames = int(checkpoint.get("total_frames", 0) or 0)
        fps = float(checkpoint.get("fps", 0.0) or 0.0)
        extraction_complete = bool(
            total_frames > 0 and processed_frames >= total_frames
        )

        raw = read_landmark_parts(folder)
        reconstructed = read_reconstructed(reconstructed_root / video_id)

        if raw.empty:
            video_rows.append({
                "participant": participant,
                "video_id": video_id,
                "file_name": file_name,
                "processed_frames": processed_frames,
                "total_frames": total_frames,
                "fps": fps,
                "extraction_complete": extraction_complete,
                "frames_any_hand": 0,
                "frames_two_hands": 0,
                "any_hand_fraction_of_processed": 0.0,
                "world_xyz_fraction": 0.0,
                "left_detected_frames": 0,
                "right_detected_frames": 0,
                "left_longest_continuous_sec": 0.0,
                "right_longest_continuous_sec": 0.0,
                "identity_rows_present": False,
                "video_qc_status": "FAIL",
                "review_note": "No extracted CAE landmark rows",
            })
            continue

        raw["frame_index"] = pd.to_numeric(raw["frame_index"], errors="coerce")
        raw["hand_index"] = pd.to_numeric(raw["hand_index"], errors="coerce")
        detected = raw[["frame_index", "hand_index"]].dropna().drop_duplicates()

        per_frame = detected.groupby("frame_index").size()
        frames_any_hand = int(per_frame.index.nunique())
        frames_two_hands = int((per_frame >= 2).sum())
        denom = max(processed_frames, 1)

        world = raw[["world_x", "world_y", "world_z"]].apply(
            pd.to_numeric, errors="coerce"
        )
        world_fraction = float(world.notna().all(axis=1).mean())

        left_frames = right_frames = 0
        left_run_sec = right_run_sec = 0.0

        if not reconstructed.empty and "anatomical_hand" in reconstructed.columns:
            reconstructed["frame_index"] = pd.to_numeric(
                reconstructed["frame_index"], errors="coerce"
            )
            hand_frames = reconstructed[
                ["frame_index", "anatomical_hand"]
            ].dropna().drop_duplicates()

            left_idx = hand_frames[
                hand_frames["anatomical_hand"].astype(str).eq("left")
            ]["frame_index"].dropna().to_numpy()
            right_idx = hand_frames[
                hand_frames["anatomical_hand"].astype(str).eq("right")
            ]["frame_index"].dropna().to_numpy()

            left_frames = int(len(np.unique(left_idx)))
            right_frames = int(len(np.unique(right_idx)))

            if fps > 0:
                left_run_sec = longest_run(left_idx) / fps
                right_run_sec = longest_run(right_idx) / fps

        # This is recovery-candidate QC, not final attempt-level QC.
        # Lead-in/out can legitimately lower whole-video hand fraction.
        if not extraction_complete:
            status = "WARN"
            note = "Extraction incomplete; resume before final recovery use"
        elif world_fraction < 0.90:
            status = "WARN"
            note = "World XYZ coverage below 90%"
        elif left_frames == 0 and right_frames == 0:
            status = "FAIL"
            note = "No reconstructed anatomical hand tracks"
        else:
            status = "PASS"
            note = ""

        video_rows.append({
            "participant": participant,
            "video_id": video_id,
            "file_name": file_name,
            "processed_frames": processed_frames,
            "total_frames": total_frames,
            "fps": fps,
            "extraction_complete": extraction_complete,
            "frames_any_hand": frames_any_hand,
            "frames_two_hands": frames_two_hands,
            "any_hand_fraction_of_processed": frames_any_hand / denom,
            "two_hand_fraction_of_processed": frames_two_hands / denom,
            "world_xyz_fraction": world_fraction,
            "left_detected_frames": left_frames,
            "right_detected_frames": right_frames,
            "left_longest_continuous_sec": left_run_sec,
            "right_longest_continuous_sec": right_run_sec,
            "identity_rows_present": not reconstructed.empty,
            "video_qc_status": status,
            "review_note": note,
        })

    video_qc = pd.DataFrame(video_rows)

    recovery_rows = []
    for _, target in targets.iterrows():
        participant = str(target["participant"])
        hand = str(target["missing_hand"])
        candidates = video_qc[video_qc["participant"].eq(participant)]

        if candidates.empty:
            recovery_rows.append({
                **target.to_dict(),
                "cae_hand_video_found": False,
                "cae_video_ids": "",
                "cae_files": "",
                "cae_detected_frames_for_missing_hand": 0,
                "cae_longest_continuous_sec_for_missing_hand": 0.0,
                "cae_world_xyz_fraction": 0.0,
                "cae_extraction_complete": False,
                "recovery_candidate_status": "NO_CAE_VIDEO",
                "next_action": "No CAE-HAND video found; retain as missing",
            })
            continue

        frame_col = f"{hand}_detected_frames"
        run_col = f"{hand}_longest_continuous_sec"

        best = candidates.sort_values(
            [frame_col, run_col],
            ascending=False,
        ).iloc[0]

        hand_frames = int(best.get(frame_col, 0))
        hand_run = float(best.get(run_col, 0.0))
        world_fraction = float(best.get("world_xyz_fraction", 0.0))
        complete = bool(best.get("extraction_complete", False))

        if not complete:
            candidate_status = "INCOMPLETE"
            next_action = "Resume CAE extraction"
        elif hand_frames < max(300, int(float(best.get("fps", 0) or 0) * 10)):
            candidate_status = "INSUFFICIENT_HAND_TRACK"
            next_action = "Visually review CAE-HAND; quantitative salvage not yet supported"
        elif hand_run < 1.0:
            candidate_status = "FRAGMENTED_TRACK"
            next_action = "Visually review tracking continuity"
        elif world_fraction < 0.90:
            candidate_status = "WORLD_XYZ_WARN"
            next_action = "Review coordinate quality before quantitative salvage"
        else:
            candidate_status = "RECOVERY_CANDIDATE"
            next_action = (
                "Map video-local attempt windows, then run motion/grip metrics"
            )

        recovery_rows.append({
            **target.to_dict(),
            "cae_hand_video_found": True,
            "cae_video_ids": ";".join(candidates["video_id"].astype(str)),
            "cae_files": ";".join(candidates["file_name"].astype(str)),
            "cae_detected_frames_for_missing_hand": hand_frames,
            "cae_longest_continuous_sec_for_missing_hand": hand_run,
            "cae_world_xyz_fraction": world_fraction,
            "cae_extraction_complete": complete,
            "recovery_candidate_status": candidate_status,
            "next_action": next_action,
        })

    recovery = pd.DataFrame(recovery_rows)
    return video_qc, recovery


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Identify missing/suspicious GoPro camera sides and process only "
            "the corresponding CAE-HAND videos as a secondary recovery source."
        )
    )
    parser.add_argument("--gopro-inventory", required=True)
    parser.add_argument("--mapping-v2", required=True)
    parser.add_argument("--cae-root", required=True)
    parser.add_argument("--cae-output-root", required=True)
    parser.add_argument(
        "--sidecar-dir",
        default=str(Path(__file__).resolve().parent),
        help="Folder containing 01_cae_inventory.py, 02_cae_pilot_extract.py and 04_cae_reconstruct_hands.py",
    )
    parser.add_argument(
        "--duration-ratio-threshold",
        type=float,
        default=0.60,
        help=(
            "Flag a camera as suspicious when its total GoPro duration is "
            "less than this fraction of the opposite camera. Default: 0.60."
        ),
    )
    parser.add_argument(
        "--manual-target",
        action="append",
        type=parse_manual_target,
        default=[],
        help="Extra known missing/partial camera, e.g. --manual-target P18:left",
    )
    parser.add_argument(
        "--only-participant",
        action="append",
        default=[],
        help="Optional restriction, e.g. --only-participant P18",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help=(
            "Actually open/download/process target CAE-HAND videos. "
            "Without --run the script only writes the recovery plan."
        ),
    )
    args = parser.parse_args()

    gopro_inventory_path = Path(args.gopro_inventory)
    mapping_v2_path = Path(args.mapping_v2)
    cae_root = Path(args.cae_root)
    output_root = Path(args.cae_output_root)
    sidecar_dir = Path(args.sidecar_dir)

    output_root.mkdir(parents=True, exist_ok=True)
    recovery_dir = output_root / "recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)

    gopro_inventory = pd.read_csv(gopro_inventory_path, dtype=str).fillna("")
    mapping_v2 = pd.read_csv(mapping_v2_path, dtype=str).fillna("")

    targets = build_recovery_targets(
        gopro_inventory,
        mapping_v2,
        args.duration_ratio_threshold,
        args.manual_target,
    )

    if args.only_participant:
        wanted = {p.upper() for p in args.only_participant}
        targets = targets[targets["participant"].str.upper().isin(wanted)]

    target_path = recovery_dir / "recovery_targets.csv"
    targets.to_csv(target_path, index=False)

    print("=== CAE SECONDARY RECOVERY PLAN ===")
    print(f"Wrote: {target_path}")
    if targets.empty:
        print("No missing/suspicious GoPro camera sides detected.")
        return

    print()
    print(targets.to_string(index=False))

    participants = sorted(
        targets["participant"].astype(str).unique(),
        key=participant_number,
    )

    print()
    print("Target CAE-HAND participants:")
    print(" ".join(participants))

    if not args.run:
        print()
        print("PLAN ONLY. No CAE video was opened.")
        print("Review recovery_targets.csv, then rerun with --run.")
        return

    required_scripts = [
        sidecar_dir / "01_cae_inventory.py",
        sidecar_dir / "02_cae_pilot_extract.py",
        sidecar_dir / "04_cae_reconstruct_hands.py",
    ]
    missing_scripts = [str(p) for p in required_scripts if not p.exists()]
    if missing_scripts:
        raise SystemExit(
            "Missing sidecar script<SET_YOUR_ANALYSIS_ROOT>" + "\n".join(missing_scripts)
        )

    python = sys.executable

    # Inventory ONLY the recovery participants. Do not --probe here:
    # opening the video would hydrate all cloud files before extraction.
    run_command([
        python,
        str(sidecar_dir / "01_cae_inventory.py"),
        "--cae-root", str(cae_root),
        "--output-root", str(output_root),
        "--participants", *participants,
        "--include-room-metadata",
    ])

    cae_inventory_path = output_root / "inventory" / "cae_video_inventory.csv"

    # Full/resumable extraction. Existing P18 pilot checkpoints will resume.
    run_command([
        python,
        str(sidecar_dir / "02_cae_pilot_extract.py"),
        "--inventory", str(cae_inventory_path),
        "--output-root", str(output_root),
        "--participants", *participants,
    ])

    # Reconstruct both anatomical hands.
    run_command([
        python,
        str(sidecar_dir / "04_cae_reconstruct_hands.py"),
        "--inventory", str(cae_inventory_path),
        "--landmarks-root", str(output_root / "landmarks"),
        "--output-root", str(output_root),
        "--participants", *participants,
    ])

    cae_inventory = pd.read_csv(cae_inventory_path, dtype=str).fillna("")
    video_qc, recovery = build_recovery_qc(
        targets,
        cae_inventory,
        output_root,
    )

    video_qc_path = recovery_dir / "cae_recovery_video_qc.csv"
    recovery_path = recovery_dir / "cae_recovery_candidates.csv"

    video_qc.to_csv(video_qc_path, index=False)
    recovery.to_csv(recovery_path, index=False)

    print()
    print("=== CAE RECOVERY QC ===")
    print(f"Wrote: {video_qc_path}")
    print(f"Wrote: {recovery_path}")
    print()

    if not recovery.empty:
        cols = [
            "participant",
            "missing_hand",
            "target_reason",
            "cae_hand_video_found",
            "cae_detected_frames_for_missing_hand",
            "cae_longest_continuous_sec_for_missing_hand",
            "cae_world_xyz_fraction",
            "recovery_candidate_status",
            "next_action",
        ]
        print(recovery[cols].to_string(index=False))

    print()
    print("IMPORTANT:")
    print(
        "RECOVERY_CANDIDATE means the secondary CAE hand track exists and "
        "is suitable for attempt mapping. It does NOT yet mean CAE is "
        "numerically interchangeable with GoPro."
    )
    print(
        "Next: annotate video-local attempt windows for recovery candidates, "
        "then run source-specific motion and grip metrics."
    )


if __name__ == "__main__":
    main()
