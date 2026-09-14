from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str).fillna("")


def as_float(v, default=float("nan")):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def pnum(v):
    m = re.search(r"(\d+)", str(v))
    return int(m.group(1)) if m else 999999


def split_semicolon(v):
    return [x for x in str(v).split(";") if x]


def source_for_hand(hand):
    return f"gopro_{str(hand).lower()}"


def group_meta(qrow):
    files = split_semicolon(qrow.get("recommended_video_files", ""))
    paths = split_semicolon(qrow.get("recommended_video_paths", ""))
    ids = split_semicolon(qrow.get("recommended_video_ids", ""))
    if len(files) != len(paths):
        return None

    items = []
    prefix = 0.0
    for i, (file_name, path) in enumerate(zip(files, paths)):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if fps <= 0 or frames <= 0:
            return None
        duration = frames / fps
        items.append({
            "video_id": ids[i] if i < len(ids) else "",
            "file_name": file_name,
            "path": path,
            "prefix": prefix,
            "duration": duration,
        })
        prefix += duration
    return items, prefix


def locate_virtual(t, group):
    for i, item in enumerate(group):
        start = item["prefix"]
        end = start + item["duration"]
        if start <= t < end or (i == len(group)-1 and math.isclose(t, end)):
            return i, item, t - start
    return None, None, None


def queue_lookup(queue):
    return {
        (r["participant"], r["block_id"], r["source"]): r
        for _, r in queue.iterrows()
    }


def bootstrap_lookup(bootstrap):
    cols = [
        "participant","block_id","attempt_id","attempt_sequence","round","device",
        "reference_start_sec","reference_end_sec"
    ]
    return (
        bootstrap[cols]
        .drop_duplicates("attempt_id")
        .set_index("attempt_id")
    )


def fragment_lookup(fragment):
    if fragment.empty:
        return {}
    return {
        (r["queue_id"], r["attempt_id"]): r
        for _, r in fragment.iterrows()
    }


def base_validated_row(base, source, qid):
    out = base.copy()
    out.update({
        "chosen_source": source,
        "coverage": "complete",
        "mapping_status": "validated",
        "mapping_confidence": "high",
        "review_note": f"Propagated from validated timing evidence {qid}.",
        "cross_file": False,
        "start_video_id": "",
        "end_video_id": "",
        "start_video_file": "",
        "end_video_file": "",
        "start_video_path": "",
        "end_video_path": "",
    })
    return out


def from_fragment(base, qrow, frag):
    if frag is None:
        return None, "fragment_coverage_missing"

    coverage = str(frag.get("fragment_coverage", ""))
    if coverage != "complete":
        return None, f"fragment_{coverage}"

    out = base_validated_row(base, qrow["source"], qrow["queue_id"])
    out["mapping_method"] = "validated_fragment_offset"

    start_id = str(frag.get("start_video_id", ""))
    end_id = str(frag.get("end_video_id", ""))
    start_file = str(frag.get("start_video_file", ""))
    end_file = str(frag.get("end_video_file", ""))
    slocal = str(frag.get("start_local_sec", ""))
    elocal = str(frag.get("end_local_sec", ""))

    if start_id and end_id and start_id == end_id:
        out.update({
            "selected_video_id": start_id,
            "video_file_name": start_file,
            "video_path": "",
            "local_start_sec": slocal,
            "local_end_sec": elocal,
        })
        return out, None

    # Complete attempt spanning two fragment files: preserve both.
    if start_id and end_id:
        out.update({
            "selected_video_id": f"{start_id};{end_id}",
            "video_file_name": f"{start_file};{end_file}",
            "video_path": "",
            "local_start_sec": slocal,
            "local_end_sec": elocal,
            "cross_file": True,
            "start_video_id": start_id,
            "end_video_id": end_id,
            "start_video_file": start_file,
            "end_video_file": end_file,
            "mapping_method": "validated_fragment_offset_cross_file",
        })
        return out, None

    return None, "fragment_location_missing"


def from_validated_block(base, qrow, ref_start, ref_end):
    if qrow["validation_status"] != "validated":
        return None, f"queue_{qrow['validation_status'] or 'blank'}"

    offset = as_float(qrow.get("validated_offset_sec", ""))
    if not math.isfinite(offset):
        return None, "validated_offset_missing"

    gm = group_meta(qrow)
    if gm is None:
        return None, "video_group_metadata_failed"

    group, total = gm
    vs = ref_start - offset
    ve = ref_end - offset

    if ve <= 0 or vs >= total:
        return None, "outside_source_group"
    if vs < 0 or ve > total:
        return None, "partial_source_window"

    si, sitem, slocal = locate_virtual(vs, group)
    ei, eitem, elocal = locate_virtual(ve, group)
    if sitem is None or eitem is None:
        return None, "source_location_failed"

    out = base_validated_row(base, qrow["source"], qrow["queue_id"])
    out["mapping_method"] = "validated_block_offset"

    if si == ei:
        out.update({
            "selected_video_id": sitem["video_id"],
            "video_file_name": sitem["file_name"],
            "video_path": sitem["path"],
            "local_start_sec": f"{slocal:.6f}",
            "local_end_sec": f"{elocal:.6f}",
        })
    else:
        # Keep the two source files explicitly. Metrics v2 will concatenate
        # the already-extracted landmark streams; no video re-extraction.
        out.update({
            "selected_video_id": f"{sitem['video_id']};{eitem['video_id']}",
            "video_file_name": f"{sitem['file_name']};{eitem['file_name']}",
            "video_path": f"{sitem['path']};{eitem['path']}",
            "local_start_sec": f"{slocal:.6f}",
            "local_end_sec": f"{elocal:.6f}",
            "cross_file": True,
            "start_video_id": sitem["video_id"],
            "end_video_id": eitem["video_id"],
            "start_video_file": sitem["file_name"],
            "end_video_file": eitem["file_name"],
            "start_video_path": sitem["path"],
            "end_video_path": eitem["path"],
            "mapping_method": "validated_block_offset_cross_file",
            "review_note": (
                f"Attempt spans {sitem['file_name']} -> {eitem['file_name']} "
                f"after validated block offset {qrow['queue_id']}; "
                "landmark streams must be concatenated for metrics."
            ),
        })

    return out, None


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Propagate final block timing into attempt×hand mappings. "
            "Supports validated attempts that span two consecutive GoPro files."
        )
    )
    ap.add_argument("--working-map", required=True)
    ap.add_argument("--bootstrap", required=True)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--fragment-coverage")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()

    work = read_csv(Path(args.working_map))
    boot = read_csv(Path(args.bootstrap))
    queue = read_csv(Path(args.queue))
    frag = (
        read_csv(Path(args.fragment_coverage))
        if args.fragment_coverage and Path(args.fragment_coverage).exists()
        else pd.DataFrame()
    )

    ql = queue_lookup(queue)
    bl = bootstrap_lookup(boot)
    fl = fragment_lookup(frag)

    rows = []
    audit = []

    for _, w in work.iterrows():
        aid = w["attempt_id"]
        hand = str(w["hand"]).lower() if "hand" in work.columns else str(w["camera"]).lower()
        participant = w["participant"]

        if aid not in bl.index:
            row = w.to_dict()
            row["hand"] = hand
            row["mapping_status"] = "unresolved_missing_bootstrap"
            rows.append(row)
            audit.append({
                "attempt_id": aid, "participant": participant,
                "hand": hand, "issue": "missing_bootstrap"
            })
            continue

        b = bl.loc[aid]
        block_id = b["block_id"]
        ref_start = float(b["reference_start_sec"])
        ref_end = float(b["reference_end_sec"])

        base = w.to_dict()
        base.update({
            "hand": hand,
            "block_id": block_id,
            "reference_start_sec": ref_start,
            "reference_end_sec": ref_end,
        })

        primary = ql.get((participant, block_id, source_for_hand(hand)))
        chosen = None
        issue = None

        # Primary GoPro always gets first refusal.
        if primary is not None:
            status = str(primary["validation_status"])
            if status == "validated":
                chosen, issue = from_validated_block(base, primary, ref_start, ref_end)
            elif status == "validated_fragment":
                chosen, issue = from_fragment(
                    base, primary, fl.get((primary["queue_id"], aid))
                )
            elif status == "not_usable":
                issue = "primary_not_usable"
            else:
                # Uncertain/pending primary is a hard blocker: never silently
                # replace it with CAE.
                row = base.copy()
                row.update({
                    "mapping_status": "unresolved_primary_not_final",
                    "review_note": (
                        f"Primary queue {primary['queue_id']} "
                        f"status={primary['validation_status']}"
                    ),
                })
                rows.append(row)
                audit.append({
                    "attempt_id": aid, "participant": participant,
                    "hand": hand, "issue": "primary_not_final",
                    "queue_id": primary["queue_id"],
                })
                continue

        # CAE secondary recovery only when primary is absent/not usable/not
        # complete for the attempt and CAE was explicitly queued for this hand.
        if chosen is None:
            cae = ql.get((participant, block_id, "cae_hand"))
            if cae is not None:
                targets = set(split_semicolon(cae.get("target_hands", "")))
                if hand in targets:
                    if cae["validation_status"] == "validated":
                        chosen, cae_issue = from_validated_block(
                            base, cae, ref_start, ref_end
                        )
                        if chosen is None:
                            issue = f"cae_{cae_issue}"
                    elif cae["validation_status"] not in {"not_usable", "no_source"}:
                        row = base.copy()
                        row.update({
                            "mapping_status": "unresolved_cae_not_final",
                            "review_note": (
                                f"CAE queue {cae['queue_id']} "
                                f"status={cae['validation_status']}"
                            ),
                        })
                        rows.append(row)
                        audit.append({
                            "attempt_id": aid, "participant": participant,
                            "hand": hand, "issue": "cae_not_final",
                            "queue_id": cae["queue_id"],
                        })
                        continue

        if chosen is None:
            row = base.copy()
            row.update({
                "chosen_source": "",
                "selected_video_id": "",
                "video_file_name": "",
                "video_path": "",
                "local_start_sec": "",
                "local_end_sec": "",
                "coverage": "none",
                "mapping_status": "unavailable",
                "mapping_method": "validated_source_hierarchy",
                "mapping_confidence": "high",
                "review_note": (
                    f"No complete reliably mapped source available "
                    f"({issue or 'no source'})."
                ),
                "cross_file": False,
                "start_video_id": "",
                "end_video_id": "",
            })
            rows.append(row)
        else:
            rows.append(chosen)

    out = pd.DataFrame(rows)
    out["_p"] = out["participant"].map(pnum)
    out["_seq"] = pd.to_numeric(out.get("attempt_sequence", ""), errors="coerce")
    out = (
        out.sort_values(["_p","_seq","hand"])
        .drop(columns=["_p","_seq"])
        .reset_index(drop=True)
    )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    candidate = outdir / "attempt_source_map_candidate_v2.csv"
    audit_path = outdir / "mapping_propagation_audit_v2.csv"
    out.to_csv(candidate, index=False)
    pd.DataFrame(audit).to_csv(audit_path, index=False)

    print("=== PROPAGATION V2 COMPLETE ===")
    print("Rows:", len(out))
    print("Wrote:", candidate)
    print("Wrote:", audit_path)
    print()
    print("Mapping status:")
    print(out["mapping_status"].value_counts(dropna=False).to_string())
    print()
    print("Cross-file validated rows:", int(out.get("cross_file", False).astype(str).str.lower().isin(["true","1"]).sum()))

    unresolved = out["mapping_status"].astype(str).str.startswith("unresolved")
    blockers = int(unresolved.sum())

    print("Freeze blockers:", blockers)
    if blockers:
        print("Do NOT run final metrics yet.")
        print(
            out.loc[
                unresolved,
                ["attempt_id","participant","hand","mapping_status","review_note"]
            ].head(100).to_string(index=False)
        )
        return

    if args.freeze:
        frozen = outdir / "attempt_source_map_FROZEN.csv"
        shutil.copy2(candidate, frozen)
        manifest = frozen.with_suffix(".manifest.json")
        manifest.write_text(
            json.dumps({
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "rows": len(out),
                "validated": int(out["mapping_status"].eq("validated").sum()),
                "unavailable": int(out["mapping_status"].eq("unavailable").sum()),
                "cross_file_validated": int(
                    out.get("cross_file", False)
                    .astype(str).str.lower().isin(["true","1"]).sum()
                ),
            }, indent=2),
            encoding="utf-8",
        )
        print()
        print("FROZEN:", frozen)
        print("Manifest:", manifest)
    else:
        print()
        print("Candidate map is blocker-free. Rerun with --freeze.")

if __name__ == "__main__":
    main()
