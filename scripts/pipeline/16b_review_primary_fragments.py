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


FINAL = {"validated_fragment", "not_usable"}


def pnum(v):
    m = re.search(r"(\d+)", str(v))
    return int(m.group(1)) if m else 999999


def natural_key(v):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(v))]


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str).fillna("")


def as_float(v, default=float("nan")):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def save_atomic(df, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def backup(path: Path):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = path.with_name(f"{path.stem}.backup-{stamp}{path.suffix}")
    shutil.copy2(path, dest)
    return dest


def usable_inventory(inv, participant, source):
    camera = "left" if source == "gopro_left" else "right"
    x = inv[
        inv["participant"].astype(str).eq(participant)
        & inv["camera"].astype(str).str.lower().eq(camera)
    ].copy()
    if "probe_status" in x.columns:
        x = x[x["probe_status"].eq("ok")]
    if "excluded" in x.columns:
        x = x[~x["excluded"].astype(str).str.lower().isin(["true","1","yes"])]
    x["duration"] = pd.to_numeric(x["duration_sec"], errors="coerce")
    x = x[x["duration"].notna() & (x["duration"] > 0)]
    return x.sort_values("file_name", key=lambda s: s.map(natural_key))


def group_meta(inv_rows):
    files = inv_rows.to_dict("records")
    prefixes, total = [], 0.0
    for f in files:
        prefixes.append(total)
        total += float(f["duration"])
    return files, prefixes, total


def locate_virtual(t, files, prefixes):
    for i, f in enumerate(files):
        start = prefixes[i]
        end = start + float(f["duration"])
        if start <= t < end or (i == len(files)-1 and math.isclose(t, end)):
            return {
                "file_index": i,
                "video_id": f["video_id"],
                "file_name": f["file_name"],
                "path": f["path"],
                "local_sec": t - start,
            }
    return None


def draw(frame, row, cand, file_name, cur, duration, prefix, mark, paused):
    lines = [
        f"{row['queue_id']} | {row['participant']} | {row['block_id']} | {row['source']}",
        f"CANDIDATE WORKBOOK ATTEMPT START: {cand['attempt_id']} | round={cand.get('round','')} | device={cand.get('device','')}",
        "Find the START of whichever candidate attempt you can identify visually.",
        f"video: {file_name}",
        f"file={cur:.3f}s/{duration:.3f}s | virtual={prefix+cur:.3f}s | {'PAUSED' if paused else 'PLAYING'}",
        f"MARK={'--' if mark is None else f'{mark:.3f}s file / {prefix+mark:.3f}s virtual'}",
        "a/d previous/next candidate attempt | n next video",
        "SPACE play/pause | j/l +/-5s | J/L +/-30s | ,/. frame",
        "s mark CURRENT candidate's WORKBOOK ATTEMPT START | v validate fragment",
        "x uncertain | u fragment unusable | q save+quit",
    ]
    y = 28
    for line in lines:
        cv2.putText(frame,line,(15,y),cv2.FONT_HERSHEY_SIMPLEX,.54,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(frame,line,(15,y),cv2.FONT_HERSHEY_SIMPLEX,.54,(0,0,0),1,cv2.LINE_AA)
        y += 25
    return frame


def coverage_rows(row, attempts, offset, files, prefixes, total):
    output = []
    for _, a in attempts.iterrows():
        rs = float(a["reference_start_sec"])
        re_ = float(a["reference_end_sec"])
        vs = rs - offset
        ve = re_ - offset

        if vs >= 0 and ve <= total:
            coverage = "complete"
        elif vs < 0 < ve <= total:
            coverage = "partial_start"
        elif 0 <= vs < total < ve:
            coverage = "partial_end"
        elif vs < 0 and ve > total:
            coverage = "partial_both"
        else:
            coverage = "outside_fragment"

        sl = locate_virtual(vs, files, prefixes) if 0 <= vs <= total else None
        el = locate_virtual(ve, files, prefixes) if 0 <= ve <= total else None

        output.append({
            "queue_id": row["queue_id"],
            "participant": row["participant"],
            "block_id": row["block_id"],
            "source": row["source"],
            "attempt_id": a["attempt_id"],
            "attempt_sequence": a["attempt_sequence"],
            "round": a.get("round",""),
            "device": a.get("device",""),
            "validated_offset_sec": offset,
            "virtual_start_sec": vs,
            "virtual_end_sec": ve,
            "fragment_coverage": coverage,
            "start_video_id": "" if not sl else sl["video_id"],
            "start_video_file": "" if not sl else sl["file_name"],
            "start_local_sec": "" if not sl else sl["local_sec"],
            "end_video_id": "" if not el else el["video_id"],
            "end_video_file": "" if not el else el["file_name"],
            "end_local_sec": "" if not el else el["local_sec"],
            "single_file": bool(sl and el and sl["video_id"] == el["video_id"]),
        })
    return output


def main():
    ap = argparse.ArgumentParser(description="Resolve incomplete primary GoPro fragments using one visually identified workbook-attempt start.")
    ap.add_argument("--queue", required=True)
    ap.add_argument("--bootstrap", required=True)
    ap.add_argument("--gopro-inventory", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--queue-ids", nargs="*")
    ap.add_argument("--reviewer", default="")
    args = ap.parse_args()

    qpath = Path(args.queue)
    q = read_csv(qpath)
    boot = read_csv(Path(args.bootstrap))
    inv = read_csv(Path(args.gopro_inventory))
    outpath = Path(args.output)

    mask = q["review_type"].eq("identify_primary_fragment_coverage")
    if args.queue_ids:
        mask &= q["queue_id"].isin(set(args.queue_ids))
    mask &= ~q["validation_status"].isin(FINAL)
    indices = list(q[mask].index)

    if not indices:
        print("No unresolved primary-fragment rows selected.")
        return

    print("Backup:", backup(qpath))
    existing = read_csv(outpath) if outpath.exists() else pd.DataFrame()

    for pos, idx in enumerate(indices, 1):
        row = q.loc[idx]
        participant, source = row["participant"], row["source"]

        attempts = boot[
            boot["participant"].eq(participant)
            & boot["block_id"].eq(row["block_id"])
            & boot["source"].eq(source)
        ][["attempt_id","attempt_sequence","round","device","reference_start_sec","reference_end_sec"]].drop_duplicates("attempt_id")
        attempts["seq"] = pd.to_numeric(attempts["attempt_sequence"],errors="coerce")
        attempts = attempts.sort_values("seq").reset_index(drop=True)

        inv_rows = usable_inventory(inv, participant, source)
        files, prefixes, total = group_meta(inv_rows)
        if not files:
            q.at[idx,"validation_status"]="not_usable"
            save_atomic(q,qpath)
            continue

        candidate_idx = min(len(attempts)//2, max(len(attempts)-1,0))
        file_idx = 0
        mark = None
        mark_file_idx = None
        done = False

        print(f"\n[{pos}/{len(indices)}] {row['queue_id']} {participant} {source}")
        print("Use a candidate attempt you can positively identify; do not guess.")

        while not done:
            f = files[file_idx]
            cap = cv2.VideoCapture(f["path"])
            if not cap.isOpened():
                raise RuntimeError(f"Could not open {f['path']}")
            fps=float(cap.get(cv2.CAP_PROP_FPS) or 0); frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration=frames/fps if fps>0 else float(f["duration"])
            cur=0.0; paused=True; last=None; switch=False

            while True:
                if not paused or last is None:
                    ok,frame=cap.read()
                    if not ok:
                        paused=True
                        cap.set(cv2.CAP_PROP_POS_MSEC,max(duration-.1,0)*1000)
                        last=None
                        continue
                    last=frame
                    fn=cap.get(cv2.CAP_PROP_POS_FRAMES)-1
                    cur=fn/fps if fps>0 else cap.get(cv2.CAP_PROP_POS_MSEC)/1000

                cand=attempts.iloc[candidate_idx]
                cv2.imshow("SIMETRIC primary fragment reviewer",draw(last.copy(),row,cand,f["file_name"],cur,duration,prefixes[file_idx],mark,paused))
                key=cv2.waitKeyEx(30 if not paused else 0)
                if key==-1: continue
                low=key & 0xFF

                if low==ord(" "): paused=not paused
                elif low==ord("j"): cur=max(cur-5,0);cap.set(cv2.CAP_PROP_POS_MSEC,cur*1000);last=None
                elif low==ord("l"): cur=min(cur+5,duration-.001);cap.set(cv2.CAP_PROP_POS_MSEC,cur*1000);last=None
                elif low==ord("J"): cur=max(cur-30,0);cap.set(cv2.CAP_PROP_POS_MSEC,cur*1000);last=None
                elif low==ord("L"): cur=min(cur+30,duration-.001);cap.set(cv2.CAP_PROP_POS_MSEC,cur*1000);last=None
                elif low==ord(","): cur=max(cur-(1/fps if fps else .033),0);cap.set(cv2.CAP_PROP_POS_MSEC,cur*1000);paused=True;last=None
                elif low==ord("."): cur=min(cur+(1/fps if fps else .033),duration-.001);cap.set(cv2.CAP_PROP_POS_MSEC,cur*1000);paused=True;last=None
                elif low==ord("a"):
                    candidate_idx=(candidate_idx-1)%len(attempts)
                    mark=None;mark_file_idx=None
                elif low==ord("d"):
                    candidate_idx=(candidate_idx+1)%len(attempts)
                    mark=None;mark_file_idx=None
                elif low==ord("n"):
                    file_idx=(file_idx+1)%len(files)
                    mark=None;mark_file_idx=None;switch=True;break
                elif low==ord("s"):
                    mark=cur;mark_file_idx=file_idx
                    print(f"Marked {cand['attempt_id']} start at {cur:.3f}s in {f['file_name']} (virtual {prefixes[file_idx]+cur:.3f}s)")
                elif low==ord("v"):
                    if mark is None or mark_file_idx!=file_idx:
                        print("Press s at the CURRENT candidate attempt start first.")
                        continue
                    ref=float(cand["reference_start_sec"])
                    virtual=prefixes[file_idx]+mark
                    offset=ref-virtual
                    rows=coverage_rows(row,attempts,offset,files,prefixes,total)
                    new=pd.DataFrame(rows)
                    if not existing.empty:
                        existing=existing[existing["queue_id"].ne(row["queue_id"])]
                    existing=pd.concat([existing,new],ignore_index=True)
                    outpath.parent.mkdir(parents=True,exist_ok=True)
                    existing.to_csv(outpath,index=False)

                    q.at[idx,"validation_status"]="validated_fragment"
                    q.at[idx,"validated_anchor_local_sec"]=f"{mark:.6f}"
                    q.at[idx,"validated_offset_sec"]=f"{offset:.6f}"
                    q.at[idx,"reviewer"]=args.reviewer
                    q.at[idx,"review_note"]=f"Fragment anchored using {cand['attempt_id']} start in {f['file_name']}."
                    save_atomic(q,qpath)

                    print(f"VALIDATED FRAGMENT {row['queue_id']}: offset={offset:.3f}s")
                    print(new["fragment_coverage"].value_counts().to_string())
                    done=True;break
                elif low==ord("x"):
                    q.at[idx,"validation_status"]="uncertain"
                    q.at[idx,"reviewer"]=args.reviewer
                    save_atomic(q,qpath);done=True;break
                elif low==ord("u"):
                    q.at[idx,"validation_status"]="not_usable"
                    q.at[idx,"reviewer"]=args.reviewer
                    save_atomic(q,qpath);done=True;break
                elif low==ord("q"):
                    save_atomic(q,qpath);cap.release();cv2.destroyAllWindows();print("Progress saved.");return

            cap.release()
            if switch: continue

    cv2.destroyAllWindows()
    print("Fragment review complete.")
    print("Coverage file:", outpath)

if __name__=="__main__":
    main()
