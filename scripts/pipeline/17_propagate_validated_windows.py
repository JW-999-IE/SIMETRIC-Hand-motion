from __future__ import annotations

import argparse
import math
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd


USABLE_FINAL = {"validated", "partial_start", "partial_end", "partial_both"}
QUEUE_FINAL = {"validated", "validated_fragment", "not_usable", "no_source"}


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


def participant_number(v):
    m = re.search(r"(\d+)", str(v))
    return int(m.group(1)) if m else 999999


def split_semicolon(v):
    return [x for x in str(v).split(";") if x]


def source_for_hand(hand):
    return f"gopro_{str(hand).lower()}"


def video_group_meta(files, paths):
    if len(files) != len(paths):
        return None
    result=[]
    total=0.0
    for file_name,path in zip(files,paths):
        cap=cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if fps<=0 or frames<=0:
            return None
        duration=frames/fps
        result.append({
            "file_name":file_name,
            "path":path,
            "prefix":total,
            "duration":duration,
        })
        total += duration
    return result,total


def locate(t, group):
    for item in group:
        start=item["prefix"];end=start+item["duration"]
        if start <= t < end or math.isclose(t,end):
            return item, t-start
    return None,None


def queue_lookup(queue):
    d={}
    for _,r in queue.iterrows():
        d[(r["participant"],r["block_id"],r["source"])]=r
    return d


def bootstrap_lookup(bootstrap):
    cols=["participant","block_id","attempt_id","attempt_sequence","round","device",
          "reference_start_sec","reference_end_sec"]
    return bootstrap[cols].drop_duplicates("attempt_id").set_index("attempt_id")


def fragment_lookup(fragment):
    if fragment.empty:
        return {}
    return {
        (r["queue_id"],r["attempt_id"]):r
        for _,r in fragment.iterrows()
    }


def final_row_from_source(base, qrow, ref_start, ref_end, hand, fragment_row=None):
    out=base.copy()
    source=qrow["source"]

    if qrow["validation_status"]=="not_usable":
        return None,"source_not_usable"

    if qrow["validation_status"]=="validated_fragment":
        if fragment_row is None:
            return None,"fragment_coverage_missing"
        coverage=fragment_row["fragment_coverage"]
        if coverage!="complete":
            return None,f"fragment_{coverage}"
        if str(fragment_row["single_file"]).lower() not in {"true","1"}:
            return None,"fragment_cross_file"
        out.update({
            "chosen_source":source,
            "selected_video_id":fragment_row["start_video_id"],
            "video_file_name":fragment_row["start_video_file"],
            "video_path":"",
            "local_start_sec":fragment_row["start_local_sec"],
            "local_end_sec":fragment_row["end_local_sec"],
            "coverage":"complete",
            "mapping_status":"validated",
            "mapping_method":"validated_fragment_offset",
            "mapping_confidence":"high",
            "review_note":f"Primary GoPro fragment validated via {qrow['queue_id']}.",
        })
        return out,None

    if qrow["validation_status"]!="validated":
        return None,f"queue_{qrow['validation_status'] or 'blank'}"

    offset=as_float(qrow["validated_offset_sec"])
    if not math.isfinite(offset):
        return None,"validated_offset_missing"

    files=split_semicolon(qrow["recommended_video_files"])
    paths=split_semicolon(qrow["recommended_video_paths"])
    meta=video_group_meta(files,paths)
    if meta is None:
        return None,"video_group_metadata_failed"
    group,total=meta

    vs=ref_start-offset
    ve=ref_end-offset

    if ve <= 0 or vs >= total:
        return None,"outside_source_group"

    if vs < 0 or ve > total:
        return None,"partial_source_window"

    sitem,slocal=locate(vs,group)
    eitem,elocal=locate(ve,group)
    if not sitem or not eitem:
        return None,"source_location_failed"

    if sitem["file_name"] != eitem["file_name"]:
        out.update({
            "chosen_source":source,
            "selected_video_id":"",
            "video_file_name":f"{sitem['file_name']};{eitem['file_name']}",
            "video_path":f"{sitem['path']};{eitem['path']}",
            "local_start_sec":slocal,
            "local_end_sec":elocal,
            "coverage":"complete",
            "mapping_status":"cross_file_requires_stitch",
            "mapping_method":"validated_block_offset_cross_file",
            "mapping_confidence":"high",
            "review_note":f"Attempt spans two source files after validated offset {qrow['queue_id']}.",
        })
        return out,"cross_file_requires_stitch"

    # video_id is easiest recovered by matching bootstrap candidate row
    # later; for now derive from queue's video-id list by same position.
    ids=split_semicolon(qrow["recommended_video_ids"])
    idx=files.index(sitem["file_name"])
    vid=ids[idx] if idx < len(ids) else ""

    out.update({
        "chosen_source":source,
        "selected_video_id":vid,
        "video_file_name":sitem["file_name"],
        "video_path":sitem["path"],
        "local_start_sec":f"{slocal:.6f}",
        "local_end_sec":f"{elocal:.6f}",
        "coverage":"complete",
        "mapping_status":"validated",
        "mapping_method":"validated_block_offset",
        "mapping_confidence":"high",
        "review_note":f"Propagated from validated block offset {qrow['queue_id']}.",
    })
    return out,None


def main():
    ap=argparse.ArgumentParser(description="Propagate validated block offsets to attempt × hand windows and freeze only when no unresolved mapping blockers remain.")
    ap.add_argument("--working-map",required=True)
    ap.add_argument("--bootstrap",required=True)
    ap.add_argument("--queue",required=True)
    ap.add_argument("--fragment-coverage")
    ap.add_argument("--output-dir",required=True)
    ap.add_argument("--freeze",action="store_true")
    args=ap.parse_args()

    work=read_csv(Path(args.working_map))
    boot=read_csv(Path(args.bootstrap))
    queue=read_csv(Path(args.queue))
    frag=read_csv(Path(args.fragment_coverage)) if args.fragment_coverage and Path(args.fragment_coverage).exists() else pd.DataFrame()

    ql=queue_lookup(queue)
    bl=bootstrap_lookup(boot)
    fl=fragment_lookup(frag)

    rows=[]
    audit=[]

    for _,w in work.iterrows():
        aid=w["attempt_id"]
        hand=str(w["hand"]).lower() if "hand" in work.columns else str(w["camera"]).lower()
        participant=w["participant"]

        if aid not in bl.index:
            row=w.to_dict()
            row["hand"]=hand
            row["mapping_status"]="unresolved_missing_bootstrap"
            rows.append(row)
            audit.append({"attempt_id":aid,"participant":participant,"hand":hand,"issue":"missing_bootstrap"})
            continue

        b=bl.loc[aid]
        block_id=b["block_id"]
        ref_start=float(b["reference_start_sec"])
        ref_end=float(b["reference_end_sec"])

        base=w.to_dict()
        base["hand"]=hand
        base["block_id"]=block_id
        base["reference_start_sec"]=ref_start
        base["reference_end_sec"]=ref_end

        primary_key=(participant,block_id,source_for_hand(hand))
        primary=ql.get(primary_key)
        chosen=None
        issue=None

        # 1. Primary GoPro first.
        if primary is not None:
            frag_row=fl.get((primary["queue_id"],aid))
            chosen,issue=final_row_from_source(base,primary,ref_start,ref_end,hand,frag_row)

            # An uncertain/pending PRIMARY cannot be silently bypassed.
            if chosen is None and str(primary["validation_status"]) not in {"not_usable","validated_fragment","validated"}:
                row=base.copy()
                row["mapping_status"]="unresolved_primary_not_final"
                row["review_note"]=f"Primary queue {primary['queue_id']} status={primary['validation_status']}"
                rows.append(row)
                audit.append({"attempt_id":aid,"participant":participant,"hand":hand,"issue":"primary_not_final","queue_id":primary["queue_id"]})
                continue

        # 2. If primary not usable/complete, try validated CAE recovery for target hand.
        if chosen is None:
            cae=ql.get((participant,block_id,"cae_hand"))
            if cae is not None:
                targets=set(split_semicolon(cae.get("target_hands","")))
                if hand in targets and cae["validation_status"]=="validated":
                    chosen,cae_issue=final_row_from_source(base,cae,ref_start,ref_end,hand,None)
                    if chosen is None:
                        issue=f"cae_{cae_issue}"

        # 3. If no usable source exists, preserve true missingness.
        if chosen is None:
            row=base.copy()
            row.update({
                "chosen_source":"",
                "selected_video_id":"",
                "video_file_name":"",
                "video_path":"",
                "local_start_sec":"",
                "local_end_sec":"",
                "coverage":"none",
                "mapping_status":"unavailable",
                "mapping_method":"validated_source_hierarchy",
                "mapping_confidence":"high",
                "review_note":f"No complete usable source after block validation ({issue or 'no source'}).",
            })
            rows.append(row)
            continue

        rows.append(chosen)
        if chosen.get("mapping_status")=="cross_file_requires_stitch":
            audit.append({"attempt_id":aid,"participant":participant,"hand":hand,"issue":"cross_file_requires_stitch"})

    out=pd.DataFrame(rows)
    out["_p"]=out["participant"].map(participant_number)
    seq=pd.to_numeric(out.get("attempt_sequence",""),errors="coerce")
    out["_seq"]=seq
    out=out.sort_values(["_p","_seq","hand"]).drop(columns=["_p","_seq"]).reset_index(drop=True)

    outdir=Path(args.output_dir);outdir.mkdir(parents=True,exist_ok=True)
    candidate=outdir/"attempt_source_map_candidate.csv"
    audit_path=outdir/"mapping_propagation_audit.csv"
    out.to_csv(candidate,index=False)
    pd.DataFrame(audit).to_csv(audit_path,index=False)

    print("=== PROPAGATION COMPLETE ===")
    print("Rows:",len(out))
    print("Wrote:",candidate)
    print("Wrote:",audit_path)
    print()
    print("Mapping status:")
    print(out["mapping_status"].value_counts(dropna=False).to_string())

    unresolved=out["mapping_status"].astype(str).str.startswith("unresolved")
    cross=out["mapping_status"].eq("cross_file_requires_stitch")
    blockers=int(unresolved.sum()+cross.sum())

    print()
    print("Freeze blockers:",blockers)

    if blockers:
        print("Do NOT run final metrics yet.")
        show=out[unresolved|cross][["attempt_id","participant","hand","mapping_status","review_note"]]
        print(show.head(100).to_string(index=False))
        return

    if args.freeze:
        frozen=outdir/"attempt_source_map_FROZEN.csv"
        shutil.copy2(candidate,frozen)
        manifest=frozen.with_suffix(".manifest.txt")
        manifest.write_text(
            f"created_at={datetime.now().isoformat(timespec='seconds')}\n"
            f"rows={len(out)}\n"
            f"validated={(out['mapping_status']=='validated').sum()}\n"
            f"unavailable={(out['mapping_status']=='unavailable').sum()}\n",
            encoding="utf-8"
        )
        print()
        print("FROZEN:",frozen)
        print("Manifest:",manifest)
    else:
        print()
        print("Candidate map is blocker-free. Rerun with --freeze to create FROZEN.csv.")

if __name__=="__main__":
    main()
