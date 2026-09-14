from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd

FINAL_STATUSES = {
    "validated", "partial_start", "partial_end", "partial_both",
    "not_visible", "unavailable",
}
USABLE_STATUSES = {"validated", "partial_start", "partial_end", "partial_both"}


def participant_number(value):
    m = re.search(r"(\d+)", str(value))
    return int(m.group(1)) if m else 999999


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")


def as_float(value, default=float("nan")):
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def usable_inventory_rows(inv: pd.DataFrame) -> pd.DataFrame:
    out = inv.copy()
    if "probe_status" in out.columns:
        out = out[out["probe_status"].astype(str).eq("ok")]
    if "excluded" in out.columns:
        out = out[
            ~out["excluded"].astype(str).str.lower().isin(["true", "1", "yes"])
        ]
    return out


def read_cae_completed_video_ids(cae_output_root: Path) -> set[str]:
    path = cae_output_root / "validation" / "extraction_result.csv"
    if not path.exists():
        return set()
    df = load_csv(path)
    return set(df[df["status"].eq("complete")]["video_id"].astype(str))


def candidate_record(source, row):
    return {
        "source": source,
        "video_id": str(row["video_id"]),
        "file_name": str(row["file_name"]),
        "path": str(row["path"]),
        "duration_sec": str(row.get("duration_sec", "")),
        "fps": str(row.get("fps", "")),
    }


def build_candidate_map(gopro_inventory, cae_inventory, cae_completed):
    candidates = {}

    gopro = usable_inventory_rows(gopro_inventory)
    gopro = gopro[gopro["camera"].str.lower().isin(["left", "right"])]
    for _, row in gopro.iterrows():
        participant = str(row["participant"])
        hand = str(row["camera"]).lower()
        candidates.setdefault((participant, hand), []).append(
            candidate_record(f"gopro_{hand}", row)
        )

    if not cae_inventory.empty:
        cae = cae_inventory[cae_inventory["camera_source"].eq("cae_hand")]
        for _, row in cae.iterrows():
            vid = str(row["video_id"])
            if cae_completed and vid not in cae_completed:
                continue
            participant = str(row["participant"])
            for hand in ("left", "right"):
                candidates.setdefault((participant, hand), []).append(
                    candidate_record("cae_hand", row)
                )

    for key, items in candidates.items():
        items.sort(key=lambda c: (
            0 if c["source"].startswith("gopro_") else 1,
            c["file_name"].lower()
        ))
    return candidates


def build_worklist(mapping_v2, candidates, existing_path=None):
    rows = []
    for _, row in mapping_v2.iterrows():
        hand = str(row["camera"]).lower()
        if hand not in {"left", "right"}:
            continue
        cands = candidates.get((str(row["participant"]), hand), [])
        rows.append({
            "attempt_id": row["attempt_id"],
            "participant": row["participant"],
            "attempt_sequence": row["attempt_sequence"],
            "round": row["round"],
            "device": row["device"],
            "hand": hand,
            "reference_start_ms": row["reference_start_ms"],
            "reference_end_ms": row["reference_end_ms"],
            "candidate_count": len(cands),
            "candidate_sources": ";".join(dict.fromkeys(c["source"] for c in cands)),
            "candidate_videos_json": json.dumps(cands, ensure_ascii=False),
            "chosen_source": "",
            "selected_video_id": "",
            "video_file_name": "",
            "video_path": "",
            "local_start_sec": "",
            "local_end_sec": "",
            "coverage": "",
            "mapping_status": "needs_video_review" if cands else "unavailable",
            "mapping_method": "",
            "mapping_confidence": "",
            "review_note": "" if cands else "No usable GoPro or completed CAE-HAND source",
            "annotated_at": "",
        })

    out = pd.DataFrame(rows)

    if existing_path and existing_path.exists():
        old = load_csv(existing_path)
        keep = [
            "chosen_source", "selected_video_id", "video_file_name",
            "video_path", "local_start_sec", "local_end_sec", "coverage",
            "mapping_status", "mapping_method", "mapping_confidence",
            "review_note", "annotated_at",
        ]
        old = old[["attempt_id", "hand"] + [c for c in keep if c in old.columns]]
        merged = out.merge(old, on=["attempt_id", "hand"], how="left", suffixes=("", "_old"))
        for c in keep:
            oc = f"{c}_old"
            if oc in merged.columns:
                mask = merged[oc].astype(str).ne("")
                merged.loc[mask, c] = merged.loc[mask, oc]
                merged = merged.drop(columns=oc)
        out = merged

    out["_p"] = out["participant"].map(participant_number)
    out["_seq"] = pd.to_numeric(out["attempt_sequence"], errors="coerce")
    return out.sort_values(["_p", "_seq", "hand"]).drop(columns=["_p", "_seq"]).reset_index(drop=True)


def save_atomic(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def audit(df):
    unresolved = ~df["mapping_status"].isin(FINAL_STATUSES)
    print("=== ATTEMPT-SOURCE MAPPING AUDIT ===")
    print(f"Rows: {len(df)}")
    print(df["mapping_status"].value_counts(dropna=False).to_string())
    print(f"\nFinal rows: {(~unresolved).sum()}")
    print(f"Unresolved rows: {unresolved.sum()}")
    print("\nCandidate source patterns:")
    print(df["candidate_sources"].replace("", "<none>").value_counts().to_string())
    print("\nUnresolved by participant:")
    x = (
        df.assign(unresolved=unresolved)
        .groupby("participant")
        .agg(rows=("attempt_id", "size"), unresolved=("unresolved", "sum"))
        .reset_index()
    )
    x["_p"] = x["participant"].map(participant_number)
    print(x.sort_values("_p").drop(columns="_p").to_string(index=False))


def open_video(candidate):
    cap = cv2.VideoCapture(candidate["path"])
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {candidate['path']}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frames / fps if fps > 0 else 0
    return cap, fps, duration


def seek(cap, sec, duration):
    sec = max(0.0, min(sec, max(duration - 0.001, 0)))
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    return sec


def draw_overlay(frame, row, candidate, cur, duration, start, end, paused):
    lines = [
        f"{row['attempt_id']} | {row['participant']} | {row['hand'].upper()} | {row['device']} | round {row['round']}",
        f"{candidate['source']} | {candidate['file_name']}",
        f"{cur:.3f}s / {duration:.3f}s | {'PAUSED' if paused else 'PLAYING'}",
        f"START={'--' if start is None else f'{start:.3f}s'} | END={'--' if end is None else f'{end:.3f}s'}",
        f"workbook reference ONLY: {row['reference_start_ms']} -> {row['reference_end_ms']} ms",
        "SPACE play/pause | j/l -/+5s | J/L -/+30s | ,/. frame | s start | e end",
        "v next source/video | w validated | 1/2/3 partial start/end/both | u not-visible",
        "x uncertain | r reset marks | q save+quit",
    ]
    y = 28
    for line in lines:
        cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)
        y += 25
    return frame


def commit(df, idx, candidate, start, end, status, coverage, note=""):
    if status in USABLE_STATUSES:
        if start is None or end is None:
            raise ValueError("Mark both start and end first.")
        if end <= start:
            raise ValueError("End must be after start.")
    df.at[idx, "chosen_source"] = candidate["source"]
    df.at[idx, "selected_video_id"] = candidate["video_id"]
    df.at[idx, "video_file_name"] = candidate["file_name"]
    df.at[idx, "video_path"] = candidate["path"]
    df.at[idx, "local_start_sec"] = "" if start is None else f"{start:.6f}"
    df.at[idx, "local_end_sec"] = "" if end is None else f"{end:.6f}"
    df.at[idx, "coverage"] = coverage
    df.at[idx, "mapping_status"] = status
    df.at[idx, "mapping_method"] = "video_local_manual_annotation"
    df.at[idx, "mapping_confidence"] = "high"
    df.at[idx, "review_note"] = note
    df.at[idx, "annotated_at"] = datetime.now().isoformat(timespec="seconds")

    # One CAE-HAND video shows both hands. Copy the same attempt time window
    # to the opposite hand if that row also has the same CAE video available
    # and is still unresolved. The anatomical hand tracks remain separate.
    if status in USABLE_STATUSES and candidate["source"] == "cae_hand":
        other = "right" if df.at[idx, "hand"] == "left" else "left"
        mask = df["attempt_id"].eq(df.at[idx, "attempt_id"]) & df["hand"].eq(other)
        if mask.any():
            j = df.index[mask][0]
            if df.at[j, "mapping_status"] not in FINAL_STATUSES:
                cands = json.loads(df.at[j, "candidate_videos_json"] or "[]")
                has_primary_gopro = any(c["source"] == f"gopro_{other}" for c in cands)
                has_same_cae = any(
                    c["source"] == "cae_hand" and c["video_id"] == candidate["video_id"]
                    for c in cands
                )
                # Never let a shared CAE annotation displace a potentially
                # usable primary GoPro source on the opposite hand.
                if has_same_cae and not has_primary_gopro:
                    df.at[j, "chosen_source"] = "cae_hand"
                    df.at[j, "selected_video_id"] = candidate["video_id"]
                    df.at[j, "video_file_name"] = candidate["file_name"]
                    df.at[j, "video_path"] = candidate["path"]
                    df.at[j, "local_start_sec"] = f"{start:.6f}"
                    df.at[j, "local_end_sec"] = f"{end:.6f}"
                    df.at[j, "coverage"] = coverage
                    df.at[j, "mapping_status"] = status
                    df.at[j, "mapping_method"] = "video_local_manual_annotation_shared_cae_window"
                    df.at[j, "mapping_confidence"] = "high"
                    df.at[j, "review_note"] = "Shared CAE attempt time window; anatomical hands analysed separately."
                    df.at[j, "annotated_at"] = datetime.now().isoformat(timespec="seconds")


def annotate(df, path, participants=None, hands=None, only_unresolved=True):
    indices = list(df.index)
    if participants:
        indices = [i for i in indices if df.at[i, "participant"].upper() in participants]
    if hands:
        indices = [i for i in indices if df.at[i, "hand"].lower() in hands]
    if only_unresolved:
        indices = [i for i in indices if df.at[i, "mapping_status"] not in FINAL_STATUSES]

    if not indices:
        print("Nothing to annotate.")
        return

    print(f"Rows selected: {len(indices)}")
    print("Saved after every committed row.")

    for pos, idx in enumerate(indices, 1):
        row = df.iloc[idx]
        cands = json.loads(row["candidate_videos_json"] or "[]")
        if not cands:
            df.at[idx, "mapping_status"] = "unavailable"
            df.at[idx, "coverage"] = "none"
            df.at[idx, "mapping_method"] = "inventory_no_source"
            df.at[idx, "mapping_confidence"] = "high"
            save_atomic(df, path)
            continue

        cand_idx = 0
        if row["selected_video_id"]:
            for k, c in enumerate(cands):
                if c["video_id"] == row["selected_video_id"]:
                    cand_idx = k
                    break

        start = as_float(row["local_start_sec"])
        end = as_float(row["local_end_sec"])
        start = start if math.isfinite(start) else None
        end = end if math.isfinite(end) else None

        done = False
        while not done:
            candidate = cands[cand_idx]
            cap, fps, duration = open_video(candidate)
            cur = seek(cap, start or 0.0, duration)
            paused = True
            last = None
            switch = False

            while True:
                if not paused or last is None:
                    ok, frame = cap.read()
                    if not ok:
                        paused = True
                        cur = seek(cap, max(duration - 0.1, 0), duration)
                        last = None
                        continue
                    last = frame
                    frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
                    cur = frame_no / fps if fps > 0 else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

                display = draw_overlay(last.copy(), row, candidate, cur, duration, start, end, paused)
                cv2.imshow("SIMETRIC attempt mapper", display)
                key = cv2.waitKeyEx(30 if not paused else 0)
                if key == -1:
                    continue
                low = key & 0xFF

                if low == ord(" "):
                    paused = not paused
                elif low == ord("j"):
                    cur = seek(cap, cur - 5, duration); last = None
                elif low == ord("l"):
                    cur = seek(cap, cur + 5, duration); last = None
                elif low == ord("J"):
                    cur = seek(cap, cur - 30, duration); last = None
                elif low == ord("L"):
                    cur = seek(cap, cur + 30, duration); last = None
                elif low == ord(","):
                    paused = True; cur = seek(cap, cur - (1/fps if fps else .033), duration); last = None
                elif low == ord("."):
                    paused = True; cur = seek(cap, cur + (1/fps if fps else .033), duration); last = None
                elif low == ord("s"):
                    start = cur; print(f"{row['attempt_id']} START {start:.3f}s")
                elif low == ord("e"):
                    end = cur; print(f"{row['attempt_id']} END {end:.3f}s")
                elif low == ord("r"):
                    start = end = None
                elif low == ord("v"):
                    cand_idx = (cand_idx + 1) % len(cands); switch = True; break
                elif low in [ord("w"), ord("1"), ord("2"), ord("3")]:
                    smap = {
                        ord("w"): ("validated", "complete"),
                        ord("1"): ("partial_start", "partial_start"),
                        ord("2"): ("partial_end", "partial_end"),
                        ord("3"): ("partial_both", "partial_both"),
                    }
                    status, coverage = smap[low]
                    try:
                        commit(df, idx, candidate, start, end, status, coverage)
                    except ValueError as exc:
                        print("NOT SAVED:", exc)
                        continue
                    save_atomic(df, path)
                    print(f"[{pos}/{len(indices)}] saved {row['attempt_id']} {row['hand']} {candidate['source']} {start:.3f}-{end:.3f}s")
                    done = True
                    break
                elif low == ord("u"):
                    commit(df, idx, candidate, None, None, "not_visible", "none",
                           "Attempt/hand not visible or usable in reviewed source.")
                    save_atomic(df, path)
                    print(f"[{pos}/{len(indices)}] not visible: {row['attempt_id']} {row['hand']}")
                    done = True
                    break
                elif low == ord("x"):
                    df.at[idx, "mapping_status"] = "uncertain"
                    df.at[idx, "review_note"] = "Reviewer skipped; source/window uncertain."
                    df.at[idx, "annotated_at"] = datetime.now().isoformat(timespec="seconds")
                    save_atomic(df, path)
                    done = True
                    break
                elif low == ord("q"):
                    save_atomic(df, path)
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Progress saved.")
                    return

            cap.release()
            if switch:
                continue

    cv2.destroyAllWindows()
    save_atomic(df, path)
    print("Selected annotation work complete.")


def freeze_mapping(df, source_path, output_path):
    unresolved = df[~df["mapping_status"].isin(FINAL_STATUSES)]
    if len(unresolved):
        print(f"Cannot freeze: {len(unresolved)} unresolved rows remain.")
        print(unresolved[["attempt_id","participant","hand","candidate_sources","mapping_status"]].head(100).to_string(index=False))
        raise SystemExit(2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows": len(df),
        "status_counts": df["mapping_status"].value_counts().to_dict(),
        "usable_rows": int(df["mapping_status"].isin(USABLE_STATUSES).sum()),
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Frozen mapping: {output_path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build")
    p.add_argument("--mapping-v2", required=True)
    p.add_argument("--gopro-inventory", required=True)
    p.add_argument("--cae-inventory", required=True)
    p.add_argument("--cae-output-root", required=True)
    p.add_argument("--output", required=True)

    p = sub.add_parser("audit")
    p.add_argument("--mapping", required=True)

    p = sub.add_parser("annotate")
    p.add_argument("--mapping", required=True)
    p.add_argument("--participants", nargs="*")
    p.add_argument("--hands", nargs="*", choices=["left","right"])
    p.add_argument("--all", action="store_true")

    p = sub.add_parser("freeze")
    p.add_argument("--mapping", required=True)
    p.add_argument("--output", required=True)

    args = ap.parse_args()

    if args.command == "build":
        mv2 = load_csv(Path(args.mapping_v2))
        gi = load_csv(Path(args.gopro_inventory))
        ci_path = Path(args.cae_inventory)
        ci = load_csv(ci_path) if ci_path.exists() else pd.DataFrame()
        completed = read_cae_completed_video_ids(Path(args.cae_output_root))
        candidates = build_candidate_map(gi, ci, completed)
        out_path = Path(args.output)
        work = build_worklist(mv2, candidates, out_path if out_path.exists() else None)
        if out_path.exists():
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup = out_path.with_name(f"{out_path.stem}.backup-{stamp}{out_path.suffix}")
            shutil.copy2(out_path, backup)
            print(f"Backup: {backup}")
        save_atomic(work, out_path)
        print(f"Wrote: {out_path}")
        audit(work)

    elif args.command == "audit":
        audit(load_csv(Path(args.mapping)))

    elif args.command == "annotate":
        path = Path(args.mapping)
        df = load_csv(path)
        participants = {p.upper() for p in args.participants} if args.participants else None
        hands = set(args.hands) if args.hands else None
        annotate(df, path, participants, hands, only_unresolved=not args.all)

    elif args.command == "freeze":
        path = Path(args.mapping)
        freeze_mapping(load_csv(path), path, Path(args.output))


if __name__ == "__main__":
    main()
