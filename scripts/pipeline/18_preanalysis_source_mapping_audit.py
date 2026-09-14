from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str).fillna("")


def split_ids(value: object) -> list[str]:
    return [x for x in str(value).split(";") if x]


def has_landmark_parts(folder: Path) -> bool:
    if not folder.exists():
        return False
    patterns = ("part-*.parquet", "part-*.csv.gz", "part-*.csv")
    return any(next(folder.glob(p), None) is not None for p in patterns)


def norm_name(value: object) -> str:
    return Path(str(value)).name.lower().strip()


def participant_num(value: object) -> int:
    m = re.search(r"(\d+)", str(value))
    return int(m.group(1)) if m else 999999


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Read-only pre-analysis integrity audit: confirm every mapped source "
            "has extracted landmark data and flag implausible left/right block-offset disagreements."
        )
    )
    ap.add_argument("--root", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--gopro-inventory", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--paired-offset-warning-sec",
        type=float,
        default=30.0,
        help="Flag left/right validated block offsets differing by more than this many seconds.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    mapping = read_csv(Path(args.mapping))
    queue = read_csv(Path(args.queue))
    inv = read_csv(Path(args.gopro_inventory))
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    source_rows = []
    for _, r in mapping[mapping["mapping_status"].eq("validated")].iterrows():
        source = str(r.get("chosen_source", ""))
        if not source.startswith("gopro_"):
            continue

        ids = split_ids(r.get("selected_video_id", ""))
        names = split_ids(r.get("video_file_name", ""))

        for i, vid in enumerate(ids):
            folder = root / "landmarks" / vid
            exists = has_landmark_parts(folder)
            expected_name = names[i] if i < len(names) else ""

            candidates = []
            if not exists and expected_name:
                x = inv[
                    inv.get("file_name", "").astype(str).map(norm_name).eq(norm_name(expected_name))
                ].copy()
                if "participant" in x.columns:
                    x = x[x["participant"].astype(str).eq(str(r["participant"]))]
                if "camera" in x.columns:
                    want_camera = "left" if source == "gopro_left" else "right"
                    x = x[x["camera"].astype(str).str.lower().eq(want_camera)]
                for _, c in x.iterrows():
                    cand_id = str(c.get("video_id", ""))
                    if cand_id and has_landmark_parts(root / "landmarks" / cand_id):
                        candidates.append(cand_id)

            source_rows.append({
                "attempt_id": r["attempt_id"],
                "participant": r["participant"],
                "hand": r.get("hand", ""),
                "chosen_source": source,
                "mapped_video_id": vid,
                "mapped_video_file": expected_name,
                "landmark_data_present": exists,
                "unique_inventory_repair_candidate": candidates[0] if len(set(candidates)) == 1 else "",
                "repair_candidate_count": len(set(candidates)),
                "issue": "" if exists else "MAPPED_VIDEO_ID_HAS_NO_LANDMARK_DATA",
            })

    source_audit = pd.DataFrame(source_rows)

    q = queue[
        queue["validation_status"].eq("validated")
        & queue["source"].isin(["gopro_left", "gopro_right"])
    ].copy()
    q["validated_offset_num"] = pd.to_numeric(q["validated_offset_sec"], errors="coerce")

    pair_rows = []
    for (participant, block_id), g in q.groupby(["participant", "block_id"]):
        left = g[g["source"].eq("gopro_left")]
        right = g[g["source"].eq("gopro_right")]
        if left.empty or right.empty:
            continue
        l = left.iloc[0]
        r = right.iloc[0]
        lo = float(l["validated_offset_num"])
        ro = float(r["validated_offset_num"])
        diff = abs(lo - ro)
        pair_rows.append({
            "participant": participant,
            "block_id": block_id,
            "left_queue_id": l["queue_id"],
            "right_queue_id": r["queue_id"],
            "left_offset_sec": lo,
            "right_offset_sec": ro,
            "absolute_difference_sec": diff,
            "warning_threshold_sec": args.paired_offset_warning_sec,
            "issue": (
                "PAIRED_CAMERA_OFFSET_DISAGREEMENT"
                if diff > args.paired_offset_warning_sec
                else ""
            ),
        })

    pair_audit = pd.DataFrame(pair_rows)
    if not pair_audit.empty:
        pair_audit["_p"] = pair_audit["participant"].map(participant_num)
        pair_audit = pair_audit.sort_values(
            ["issue", "absolute_difference_sec", "_p"],
            ascending=[False, False, True]
        ).drop(columns="_p")

    source_path = outdir / "source_landmark_integrity_audit.csv"
    pair_path = outdir / "paired_camera_offset_audit.csv"
    source_audit.to_csv(source_path, index=False)
    pair_audit.to_csv(pair_path, index=False)

    missing = source_audit[~source_audit["landmark_data_present"]]
    offset_bad = pair_audit[pair_audit["issue"].ne("")] if not pair_audit.empty else pair_audit

    print("=== PRE-ANALYSIS SOURCE / MAPPING AUDIT ===")
    print(f"Validated GoPro source components checked: {len(source_audit)}")
    print(f"Missing extracted landmark components: {len(missing)}")
    print(f"Paired camera offset warnings: {len(offset_bad)}")
    print()
    print("Wrote:", source_path)
    print("Wrote:", pair_path)

    if len(missing):
        print("\n=== MISSING SOURCE COMPONENTS ===")
        print(
            missing[
                [
                    "participant","attempt_id","hand","mapped_video_id",
                    "mapped_video_file","repair_candidate_count",
                    "unique_inventory_repair_candidate"
                ]
            ].drop_duplicates().to_string(index=False)
        )

    if len(offset_bad):
        print("\n=== OFFSET WARNINGS ===")
        print(
            offset_bad[
                [
                    "participant","block_id","left_queue_id","right_queue_id",
                    "left_offset_sec","right_offset_sec","absolute_difference_sec"
                ]
            ].to_string(index=False)
        )

    hard_blockers = len(missing)
    review_warnings = len(offset_bad)
    print()
    print("Hard source-data blockers:", hard_blockers)
    print("Paired-camera review warnings:", review_warnings)
    if hard_blockers:
        print("STOP: mapped source IDs without landmark data must be repaired before recomputing final metrics.")
    elif review_warnings:
        print(
            "SOURCE DATA PASS, WITH REVIEW WARNINGS: left/right cameras may have different recording start times, "
            "so offset disagreement is not automatically an error. Re-check only extreme rows if the visual anchor "
            "was uncertain; otherwise document the camera-clock difference and proceed."
        )
    else:
        print("PASS: source IDs resolve and there are no configured paired-camera review warnings.")


if __name__ == "__main__":
    main()
