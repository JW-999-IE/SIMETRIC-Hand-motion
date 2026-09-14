from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str).fillna("")


def split_ids(value: object) -> list[str]:
    return [x for x in str(value).split(";") if x]


def join_ids(values: list[str]) -> str:
    return ";".join(values)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Safely create a repaired frozen-mapping candidate from the "
            "read-only source integrity audit. Only unique exact repair candidates are applied."
        )
    )
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--source-audit", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    mapping_path = Path(args.mapping)
    mapping = read_csv(mapping_path)
    audit = read_csv(Path(args.source_audit))

    bad = audit[
        audit["issue"].eq("MAPPED_VIDEO_ID_HAS_NO_LANDMARK_DATA")
    ].copy()

    if bad.empty:
        print("No source-ID repairs are required.")
        return

    unresolved = bad[
        bad["repair_candidate_count"].astype(str).ne("1")
        | bad["unique_inventory_repair_candidate"].eq("")
    ]
    if len(unresolved):
        print("STOP: not every missing source has one unique repair candidate.")
        print(
            unresolved[
                [
                    "participant","attempt_id","hand","mapped_video_id",
                    "mapped_video_file","repair_candidate_count",
                    "unique_inventory_repair_candidate"
                ]
            ].drop_duplicates().to_string(index=False)
        )
        raise SystemExit(2)

    repair_map = {}
    for _, r in bad.iterrows():
        old = r["mapped_video_id"]
        new = r["unique_inventory_repair_candidate"]
        if old in repair_map and repair_map[old] != new:
            raise RuntimeError(
                f"Conflicting repair candidates for {old}: "
                f"{repair_map[old]} vs {new}"
            )
        repair_map[old] = new

    out = mapping.copy()
    changed_rows = []

    for idx, row in out.iterrows():
        ids = split_ids(row.get("selected_video_id", ""))
        if not ids:
            continue
        new_ids = [repair_map.get(x, x) for x in ids]
        if new_ids != ids:
            out.at[idx, "selected_video_id"] = join_ids(new_ids)

            # Keep cross-file explicit component IDs consistent when present.
            if "start_video_id" in out.columns:
                old = out.at[idx, "start_video_id"]
                if old in repair_map:
                    out.at[idx, "start_video_id"] = repair_map[old]
            if "end_video_id" in out.columns:
                old = out.at[idx, "end_video_id"]
                if old in repair_map:
                    out.at[idx, "end_video_id"] = repair_map[old]

            note = str(out.at[idx, "review_note"]) if "review_note" in out.columns else ""
            extra = (
                " Source video_id repaired from exact inventory/file-name match "
                "after landmark-directory integrity audit."
            )
            if "review_note" in out.columns:
                out.at[idx, "review_note"] = (note + extra).strip()

            changed_rows.append({
                "attempt_id": row["attempt_id"],
                "participant": row["participant"],
                "hand": row.get("hand", ""),
                "old_selected_video_id": join_ids(ids),
                "new_selected_video_id": join_ids(new_ids),
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    manifest = output.with_suffix(".repair_manifest.csv")
    pd.DataFrame(changed_rows).to_csv(manifest, index=False)

    print("=== SOURCE-ID REPAIR CANDIDATE CREATED ===")
    print("Changed mapping rows:", len(changed_rows))
    print("Output:", output)
    print("Manifest:", manifest)
    print()
    print("Original mapping was NOT modified.")
    print("Run script 18 again against this candidate before using it for metrics.")


if __name__ == "__main__":
    main()
