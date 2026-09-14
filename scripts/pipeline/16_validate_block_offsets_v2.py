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

FINAL = {"validated", "not_usable", "no_source"}

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

def as_int(v, default=999999):
    try:
        return int(float(v))
    except Exception:
        return default

def pnum(v):
    m = re.search(r"(\d+)", str(v))
    return int(m.group(1)) if m else 999999

def split_semicolon(v):
    return [x for x in str(v).split(";") if x]

def save_atomic(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def backup(path: Path):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = path.with_name(f"{path.stem}.backup-{stamp}{path.suffix}")
    shutil.copy2(path, dest)
    return dest

def open_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frames / fps if fps > 0 else 0.0
    return cap, fps, duration

def group_prefixes(paths: list[str]) -> tuple[list[float], list[float]]:
    durations, prefixes = [], []
    total = 0.0
    for path in paths:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video metadata: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if fps <= 0 or frames <= 0:
            raise RuntimeError(f"Invalid video metadata: {path}")
        d = frames / fps
        prefixes.append(total)
        durations.append(d)
        total += d
    return prefixes, durations

def seek(cap, sec, duration):
    sec = max(0.0, min(float(sec), max(duration - 0.001, 0.0)))
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
    return sec

def plausible_local_range(row, prefix, duration):
    ref = as_float(row.get("anchor_reference_start_sec", ""))
    omin = as_float(row.get("offset_min_sec", ""))
    omax = as_float(row.get("offset_max_sec", ""))
    if not all(math.isfinite(x) for x in (ref, omin, omax)):
        return None
    # virtual local = reference - offset
    vmin = ref - omax
    vmax = ref - omin
    lo = max(0.0, vmin - prefix)
    hi = min(duration, vmax - prefix)
    if hi < lo:
        return None
    return lo, hi

def draw(frame, row, file_name, cur, duration, prefix, mark, paused):
    rng = plausible_local_range(row, prefix, duration)
    rng_text = "not expected in this file"
    if rng:
        rng_text = f"{rng[0]:.1f}-{rng[1]:.1f}s"

    virtual_cur = prefix + cur
    marked_virtual = None if mark is None else prefix + mark

    lines = [
        f"{row['queue_id']} | {row['participant']} | {row['block_id']} | {row['source']}",
        f"ANCHOR = WORKBOOK ATTEMPT START of {row['anchor_attempt_id']}",
        "NOT first insertion movement; NOT insertion end",
        f"video: {file_name}",
        f"file time={cur:.3f}s / {duration:.3f}s | virtual group time={virtual_cur:.3f}s | {'PAUSED' if paused else 'PLAYING'}",
        f"plausible anchor range in THIS file: {rng_text}",
        f"bootstrap local={row.get('anchor_bootstrap_local_sec','')}s | uncertainty={row.get('offset_uncertainty_sec','')}s",
        f"reference attempt start={row.get('anchor_reference_start_sec','')}s",
        f"MARK={'--' if mark is None else f'{mark:.3f}s file / {marked_virtual:.3f}s virtual'}",
        "SPACE play/pause | j/l +/-5s | J/L +/-30s | ,/. frame | [/] plausible-range edges",
        "s mark WORKBOOK ATTEMPT START | v validate | n next file | x uncertain | u unusable | q quit",
    ]
    y = 27
    for line in lines:
        cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0,0,0), 1, cv2.LINE_AA)
        y += 24
    return frame

def audit(df):
    print("=== BLOCK OFFSET VALIDATION STATUS ===")
    print(f"Queue rows: {len(df)}")
    print(df["validation_status"].replace("", "<blank>").value_counts().to_string())
    unresolved = ~df["validation_status"].isin(FINAL)
    print(f"\nResolved: {(~unresolved).sum()}")
    print(f"Remaining: {unresolved.sum()}")

def select_indices(df, queue_ids, participants, priorities, sources, include_resolved):
    mask = pd.Series(True, index=df.index)
    if queue_ids:
        mask &= df["queue_id"].isin(queue_ids)
    if participants:
        mask &= df["participant"].str.upper().isin(participants)
    if priorities:
        mask &= df["priority"].astype(str).isin(priorities)
    if sources:
        mask &= df["source"].isin(sources)
    if not include_resolved:
        mask &= ~df["validation_status"].isin(FINAL)
    x = df[mask].copy()
    x["_p"] = x["participant"].map(pnum)
    x["_b"] = x["block_index"].map(as_int)
    x["_prio"] = x["priority"].map(as_int)
    return list(x.sort_values(["_prio","_p","_b","source"]).index)

def validate_row(df, idx, marked_file_sec, marked_virtual_sec, reviewer):
    row = df.loc[idx]
    ref = as_float(row.get("anchor_reference_start_sec", ""))
    if not math.isfinite(ref):
        raise ValueError("Missing anchor_reference_start_sec")

    offset = ref - marked_virtual_sec
    bstart_ref = as_float(row.get("reference_min_start_sec", ""))
    bend_ref = as_float(row.get("reference_max_end_sec", ""))

    df.at[idx, "validated_anchor_local_sec"] = f"{marked_file_sec:.6f}"
    if "validated_anchor_virtual_sec" not in df.columns:
        df["validated_anchor_virtual_sec"] = ""
    df.at[idx, "validated_anchor_virtual_sec"] = f"{marked_virtual_sec:.6f}"
    df.at[idx, "validated_offset_sec"] = f"{offset:.6f}"
    df.at[idx, "validated_block_start_local_sec"] = (
        "" if not math.isfinite(bstart_ref) else f"{bstart_ref - offset:.6f}"
    )
    df.at[idx, "validated_block_end_local_sec"] = (
        "" if not math.isfinite(bend_ref) else f"{bend_ref - offset:.6f}"
    )
    df.at[idx, "validation_status"] = "validated"
    df.at[idx, "reviewer"] = reviewer
    if "validated_at" not in df.columns:
        df["validated_at"] = ""
    df.at[idx, "validated_at"] = datetime.now().isoformat(timespec="seconds")

    omin = as_float(row.get("offset_min_sec", ""))
    omax = as_float(row.get("offset_max_sec", ""))
    warning = ""
    if all(math.isfinite(x) for x in (omin, omax)) and not (omin <= offset <= omax):
        warning = f"Validated offset {offset:.3f}s outside bootstrap [{omin:.3f},{omax:.3f}]s"
        old = str(df.at[idx, "review_note"])
        df.at[idx, "review_note"] = (old + " " + warning).strip()
    return offset, warning

def interactive(df, path, indices, reviewer):
    if not indices:
        print("Nothing selected.")
        return
    print(f"Selected rows: {len(indices)}")
    print("Saved after every decision.")

    for pos, idx in enumerate(indices, 1):
        row = df.loc[idx]

        if row.get("review_type","") == "identify_primary_fragment_coverage":
            print(f"Skipping fragment row {row['queue_id']} — use fragment reviewer later.")
            continue

        files = split_semicolon(row.get("recommended_video_files",""))
        paths = split_semicolon(row.get("recommended_video_paths",""))
        if not files or len(files) != len(paths):
            print(f"{row['queue_id']}: no usable video path list")
            df.at[idx, "validation_status"] = "uncertain"
            save_atomic(df, path)
            continue

        prefixes, durations = group_prefixes(paths)
        anchor_file = row.get("anchor_video_file","")
        file_idx = files.index(anchor_file) if anchor_file in files else 0

        mark = None
        mark_file_idx = None
        done = False

        print(f"\n[{pos}/{len(indices)}] {row['queue_id']} {row['participant']} {row['block_id']} {row['source']}")
        print(f"Anchor ATTEMPT START: {row['anchor_attempt_id']}")

        while not done:
            file_name, video_path = files[file_idx], paths[file_idx]
            cap, fps, duration = open_video(video_path)
            prefix = prefixes[file_idx]

            if file_name == anchor_file:
                start = as_float(row.get("review_seek_start_sec",""))
                if not math.isfinite(start):
                    start = as_float(row.get("anchor_bootstrap_local_sec",""), 0.0)
            else:
                start = 0.0

            cur = seek(cap, max(start,0.0), duration)
            paused = True
            last = None
            switch = False

            while True:
                if not paused or last is None:
                    ok, frame = cap.read()
                    if not ok:
                        paused = True
                        cur = seek(cap, max(duration-0.1,0), duration)
                        last = None
                        continue
                    last = frame
                    fn = cap.get(cv2.CAP_PROP_POS_FRAMES)-1
                    cur = fn/fps if fps>0 else cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0

                cv2.imshow("SIMETRIC block offset validator v2", draw(last.copy(), row, file_name, cur, duration, prefix, mark, paused))
                key = cv2.waitKeyEx(30 if not paused else 0)
                if key == -1:
                    continue
                low = key & 0xFF

                if low == ord(" "):
                    paused = not paused
                elif low == ord("j"):
                    cur = seek(cap, cur-5, duration); last=None
                elif low == ord("l"):
                    cur = seek(cap, cur+5, duration); last=None
                elif low == ord("J"):
                    cur = seek(cap, cur-30, duration); last=None
                elif low == ord("L"):
                    cur = seek(cap, cur+30, duration); last=None
                elif low == ord(","):
                    paused=True; cur=seek(cap,cur-(1/fps if fps else .033),duration); last=None
                elif low == ord("."):
                    paused=True; cur=seek(cap,cur+(1/fps if fps else .033),duration); last=None
                elif low == ord("["):
                    rng = plausible_local_range(row, prefix, duration)
                    if rng:
                        cur=seek(cap,rng[0],duration); last=None
                elif low == ord("]"):
                    rng = plausible_local_range(row, prefix, duration)
                    if rng:
                        cur=seek(cap,rng[1],duration); last=None
                elif low == ord("s"):
                    mark = cur
                    mark_file_idx = file_idx
                    print(f"Marked WORKBOOK ATTEMPT START at {mark:.3f}s in {file_name}; virtual={prefix+mark:.3f}s")
                elif low == ord("n"):
                    file_idx = (file_idx+1)%len(files)
                    mark = None
                    mark_file_idx = None
                    switch=True
                    break
                elif low == ord("v"):
                    if mark is None or mark_file_idx != file_idx:
                        print("Press s on the actual workbook ATTEMPT START first.")
                        continue
                    offset, warning = validate_row(df, idx, mark, prefix+mark, reviewer)
                    save_atomic(df, path)
                    print(f"VALIDATED {row['queue_id']}: offset={offset:.3f}s")
                    if warning:
                        print("WARNING:", warning)
                    done=True
                    break
                elif low == ord("x"):
                    df.at[idx,"validation_status"]="uncertain"
                    df.at[idx,"reviewer"]=reviewer
                    save_atomic(df,path)
                    done=True
                    break
                elif low == ord("u"):
                    df.at[idx,"validation_status"]="not_usable"
                    df.at[idx,"reviewer"]=reviewer
                    save_atomic(df,path)
                    done=True
                    break
                elif low == ord("q"):
                    save_atomic(df,path)
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Progress saved.")
                    return

            cap.release()
            if switch:
                continue

    cv2.destroyAllWindows()
    save_atomic(df,path)
    print("Selected validation work complete.")

def main():
    ap = argparse.ArgumentParser(description="Validate block offsets using WORKBOOK ATTEMPT START on a concatenated virtual recording timeline.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit")
    a.add_argument("--queue", required=True)

    v = sub.add_parser("validate")
    v.add_argument("--queue", required=True)
    v.add_argument("--queue-ids", nargs="*")
    v.add_argument("--participants", nargs="*")
    v.add_argument("--priorities", nargs="*")
    v.add_argument("--sources", nargs="*", choices=["gopro_left","gopro_right","cae_hand"])
    v.add_argument("--reviewer", default="")
    v.add_argument("--all", action="store_true")

    args = ap.parse_args()
    path = Path(args.queue)
    df = read_csv(path)

    if args.cmd == "audit":
        audit(df)
        return

    b = backup(path)
    print(f"Backup created: {b}")

    indices = select_indices(
        df,
        set(args.queue_ids) if args.queue_ids else None,
        {x.upper() for x in args.participants} if args.participants else None,
        {str(x) for x in args.priorities} if args.priorities else None,
        set(args.sources) if args.sources else None,
        args.all,
    )
    interactive(df, path, indices, args.reviewer)

if __name__ == "__main__":
    main()
