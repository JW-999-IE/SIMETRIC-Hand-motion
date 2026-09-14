from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


KEY_GROUPS = {
    "participant": ["participant", "participant_id", "id"],
    "device": ["device", "configuration"],
    "fixation": ["fixation_count", "fixation_event_count", "fixation_event_rate", "fixation_rate", "fixations"],
    "saccade": ["saccade_count", "saccade_event_count", "saccade_event_rate", "saccade_rate", "saccades"],
    "duration": ["attempt_duration", "duration_sec", "duration", "attempt_time"],
}


def norm(s):
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")


def csv_columns(path):
    try:
        return list(pd.read_csv(path, nrows=2).columns)
    except Exception:
        return []


def xlsx_columns(path):
    try:
        xl = pd.ExcelFile(path)
        out = []
        for sheet in xl.sheet_names[:10]:
            try:
                cols = list(pd.read_excel(path, sheet_name=sheet, nrows=2).columns)
                out.append((sheet, cols))
            except Exception:
                pass
        return out
    except Exception:
        return []


def score(cols):
    nc = [norm(c) for c in cols]
    found = {}
    total = 0
    for group, aliases in KEY_GROUPS.items():
        hit = any(any(a in c for a in aliases) for c in nc)
        found[group] = hit
        total += int(hit)
    return total, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            continue
        if p.stat().st_size > 200_000_000:
            continue

        if p.suffix.lower() == ".csv":
            variants = [("", csv_columns(p))]
        else:
            variants = xlsx_columns(p)

        for sheet, cols in variants:
            if not cols:
                continue
            sc, found = score(cols)
            if sc >= 3:
                rows.append({
                    "score_0_5": sc,
                    "path": str(p),
                    "sheet": sheet,
                    **{f"has_{k}": v for k, v in found.items()},
                    "columns": " | ".join(map(str, cols)),
                })

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["score_0_5", "path"], ascending=[False, True])
    out.to_csv(args.output, index=False)

    print("=== EYE ATTEMPT-DATASET DISCOVERY ===")
    if len(out):
        print(out.head(30).to_string(index=False))
    else:
        print("No candidate with >=3 of the 5 required column groups was found.")
    print()
    print("Wrote:", args.output)
    print("Choose a file containing participant/device plus fixation, saccade and duration information.")
