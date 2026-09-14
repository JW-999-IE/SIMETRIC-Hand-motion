from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str).fillna("")


def norm(v: object) -> str:
    return str(v).replace("\\", "/").strip().lower()


def has_parts(folder: Path) -> bool:
    if not folder.exists():
        return False
    return any(
        next(folder.glob(p), None) is not None
        for p in ("part-*.parquet", "part-*.csv.gz", "part-*.csv")
    )


def metadata_summary(folder: Path) -> dict:
    meta = folder / "metadata.json"
    out = {
        "metadata_exists": meta.exists(),
        "metadata_status": "",
        "metadata_source_path": "",
        "metadata_file_name": "",
        "metadata_participant": "",
        "metadata_camera": "",
    }
    if not meta.exists():
        return out
    try:
        obj = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return out

    def first(*keys):
        for k in keys:
            if k in obj and obj[k] not in (None, ""):
                return str(obj[k])
        return ""

    out.update({
        "metadata_status": first("status", "extraction_status"),
        "metadata_source_path": first("source_path", "video_path", "path", "input_path"),
        "metadata_file_name": first("file_name", "filename", "video_file"),
        "metadata_participant": first("participant"),
        "metadata_camera": first("camera", "hand"),
    })
    return out


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Read-only diagnosis for a mapped GoPro source whose landmark data "
            "cannot be found. Checks inventory, exact landmark folder, and all "
            "landmark metadata for filename/path matches."
        )
    )
    ap.add_argument("--root", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--participant", required=True)
    ap.add_argument("--hand", choices=["left", "right"], required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    mapping = read_csv(Path(args.mapping))
    inv = read_csv(Path(args.inventory))
    participant = args.participant
    hand = args.hand.lower()

    m = mapping[
        mapping["participant"].astype(str).eq(participant)
        & mapping["hand"].astype(str).str.lower().eq(hand)
        & mapping["mapping_status"].astype(str).eq("validated")
    ].copy()

    if m.empty:
        raise RuntimeError(f"No validated mapping rows for {participant} {hand}.")

    mapped_ids = sorted({
        vid
        for value in m["selected_video_id"].astype(str)
        for vid in value.split(";")
        if vid
    })
    mapped_files = sorted({
        name
        for value in m.get("video_file_name", pd.Series(dtype=str)).astype(str)
        for name in value.split(";")
        if name
    })

    inv_p = inv[
        inv["participant"].astype(str).eq(participant)
        & inv["camera"].astype(str).str.lower().eq(hand)
    ].copy()

    if "probe_status" in inv_p.columns:
        inv_p["_probe_ok"] = inv_p["probe_status"].astype(str).eq("ok")
    else:
        inv_p["_probe_ok"] = True

    if "excluded" in inv_p.columns:
        inv_p["_included"] = ~inv_p["excluded"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
    else:
        inv_p["_included"] = True

    rows = []

    for _, r in inv_p.iterrows():
        vid = str(r.get("video_id", ""))
        folder = root / "landmarks" / vid if vid else Path("")
        rows.append({
            "match_type": "inventory_candidate",
            "participant": participant,
            "hand": hand,
            "video_id": vid,
            "file_name": r.get("file_name", ""),
            "path": r.get("path", ""),
            "probe_ok": r.get("_probe_ok", True),
            "included": r.get("_included", True),
            "landmark_folder": str(folder) if vid else "",
            "landmark_folder_exists": bool(vid and folder.exists()),
            "landmark_parts_present": bool(vid and has_parts(folder)),
            **(metadata_summary(folder) if vid else {}),
        })

    for vid in mapped_ids:
        folder = root / "landmarks" / vid
        rows.append({
            "match_type": "mapped_video_id",
            "participant": participant,
            "hand": hand,
            "video_id": vid,
            "file_name": "",
            "path": "",
            "probe_ok": "",
            "included": "",
            "landmark_folder": str(folder),
            "landmark_folder_exists": folder.exists(),
            "landmark_parts_present": has_parts(folder),
            **metadata_summary(folder),
        })

    landmarks_root = root / "landmarks"
    filename_targets = {norm(x) for x in mapped_files}
    path_targets = {
        norm(x)
        for x in inv_p.get("path", pd.Series(dtype=str)).astype(str)
        if x
    }

    metadata_matches = []
    if landmarks_root.exists():
        for folder in landmarks_root.iterdir():
            if not folder.is_dir():
                continue
            meta = metadata_summary(folder)
            meta_file = norm(meta["metadata_file_name"])
            meta_path = norm(meta["metadata_source_path"])

            why = []
            if meta_file and meta_file in filename_targets:
                why.append("metadata_filename")
            if meta_path and meta_path in path_targets:
                why.append("metadata_path")
            if meta_file and any(
                meta_file == norm(Path(x).name) for x in mapped_files
            ):
                why.append("metadata_basename")

            if why:
                metadata_matches.append({
                    "match_type": "landmark_metadata_match:" + ",".join(why),
                    "participant": participant,
                    "hand": hand,
                    "video_id": folder.name,
                    "file_name": "",
                    "path": "",
                    "probe_ok": "",
                    "included": "",
                    "landmark_folder": str(folder),
                    "landmark_folder_exists": True,
                    "landmark_parts_present": has_parts(folder),
                    **meta,
                })

    rows.extend(metadata_matches)
    out = pd.DataFrame(rows).drop_duplicates()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    print("=== MISSING SOURCE DIAGNOSTIC ===")
    print(f"Participant: {participant}")
    print(f"Hand: {hand}")
    print(f"Mapped video IDs: {', '.join(mapped_ids)}")
    print(f"Mapped files: {', '.join(mapped_files)}")
    print()
    print("Inventory candidates:")
    if inv_p.empty:
        print("  NONE")
    else:
        show_cols = [
            c for c in [
                "video_id","file_name","path","duration_sec",
                "probe_status","excluded"
            ]
            if c in inv_p.columns
        ]
        print(inv_p[show_cols].to_string(index=False))

    print()
    print("Extracted landmark metadata matches:", len(metadata_matches))
    if metadata_matches:
        print(
            pd.DataFrame(metadata_matches)[
                [
                    "video_id","landmark_parts_present",
                    "metadata_status","metadata_file_name",
                    "metadata_source_path"
                ]
            ].to_string(index=False)
        )

    exact_present = any(
        has_parts(root / "landmarks" / vid) for vid in mapped_ids
    )
    any_matching_extract = any(
        bool(r["landmark_parts_present"]) for r in metadata_matches
    )

    print()
    print("Exact mapped landmark data present:", exact_present)
    print("Alternative matching extracted data present:", any_matching_extract)
    print("Wrote:", output)

    if exact_present:
        print("RESULT: exact mapped source exists; investigate metric reader/path logic.")
    elif any_matching_extract:
        print("RESULT: extraction exists under another video_id; repair mapping after verifying the match.")
    elif not inv_p.empty:
        print("RESULT: inventory contains the source, but no extracted landmarks were found. Targeted extraction is required.")
    else:
        print("RESULT: source is absent from current inventory. Re-inventory/locate the physical video before extraction.")


if __name__ == "__main__":
    main()
