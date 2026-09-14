from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import ensure_dir, participant_number


def main():
    parser = argparse.ArgumentParser(description="Build source-aware mapping workspace.")
    parser.add_argument("--mapping-v2", required=True)
    parser.add_argument("--cae-inventory", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    v2 = pd.read_csv(args.mapping_v2, dtype=str).fillna("")
    cae = pd.read_csv(args.cae_inventory, dtype=str).fillna("")
    cae_hand = cae[cae["camera_source"].eq("cae_hand")].copy()

    rows = []

    # GoPro: one expected row per existing camera-specific v2 row.
    for _, row in v2.iterrows():
        camera = row["camera"].strip().lower()
        if camera not in {"left", "right"}:
            continue
        rows.append({
            "attempt_id": row["attempt_id"],
            "participant": row["participant"],
            "attempt_sequence": row["attempt_sequence"],
            "round": row["round"],
            "device": row["device"],
            "hand": camera,
            "source": f"gopro_{camera}",
            "available_video_ids": row.get("available_video_ids", ""),
            "available_video_files": row.get("available_video_files", ""),
            "selected_video_id": "",
            "video_file_name": "",
            "local_start_sec": "",
            "local_end_sec": "",
            "coverage": "",
            "mapping_status": "unavailable_no_video" if row.get("available_video_count", "") == "0" else "needs_video_review",
            "mapping_method": "",
            "mapping_confidence": "",
            "reference_start_ms": row.get("reference_start_ms", ""),
            "reference_end_ms": row.get("reference_end_ms", ""),
            "review_note": row.get("review_note", ""),
        })

    # CAE: each HAND video is a potential source for both anatomical hands.
    attempt_base = (
        v2.sort_values(["participant", "attempt_sequence"])
        .drop_duplicates(["attempt_id"])
    )
    cae_by_participant = {
        p: g for p, g in cae_hand.groupby("participant")
    }

    for _, attempt in attempt_base.iterrows():
        p = attempt["participant"]
        g = cae_by_participant.get(p)
        ids = ";".join(g["video_id"].astype(str)) if g is not None else ""
        names = ";".join(g["file_name"].astype(str)) if g is not None else ""
        available = bool(ids)

        for hand in ("left", "right"):
            rows.append({
                "attempt_id": attempt["attempt_id"],
                "participant": p,
                "attempt_sequence": attempt["attempt_sequence"],
                "round": attempt["round"],
                "device": attempt["device"],
                "hand": hand,
                "source": "cae_hand",
                "available_video_ids": ids,
                "available_video_files": names,
                "selected_video_id": "",
                "video_file_name": "",
                "local_start_sec": "",
                "local_end_sec": "",
                "coverage": "",
                "mapping_status": "needs_video_review" if available else "unavailable_no_video",
                "mapping_method": "",
                "mapping_confidence": "",
                "reference_start_ms": attempt.get("reference_start_ms", ""),
                "reference_end_ms": attempt.get("reference_end_ms", ""),
                "review_note": "" if available else "No CAE-HAND video inventoried",
            })

    out = pd.DataFrame(rows)
    out["_p"] = out["participant"].map(participant_number)
    out["_seq"] = pd.to_numeric(out["attempt_sequence"], errors="coerce")
    out = out.sort_values(["_p", "_seq", "hand", "source"]).drop(columns=["_p", "_seq"])

    dest = Path(args.output)
    ensure_dir(dest.parent)
    out.to_csv(dest, index=False)
    print(f"Wrote: {dest}")
    print(f"Rows: {len(out)}")
    print(out.groupby(["source", "mapping_status"]).size().to_string())


if __name__ == "__main__":
    main()
