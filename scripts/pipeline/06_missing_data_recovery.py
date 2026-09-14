from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import ensure_dir


VALID_MAPPING = {"mapped", "validated", "complete", "partial_start", "partial_end", "partial_both"}


def usable(row) -> bool:
    status = str(row.get("mapping_status", "")).strip().lower()
    start = pd.to_numeric(pd.Series([row.get("local_start_sec", "")]), errors="coerce").iloc[0]
    end = pd.to_numeric(pd.Series([row.get("local_end_sec", "")]), errors="coerce").iloc[0]
    return (
        status in VALID_MAPPING
        and pd.notna(start)
        and pd.notna(end)
        and float(end) > float(start)
        and str(row.get("selected_video_id", "")).strip() != ""
    )


def main():
    parser = argparse.ArgumentParser(description="Create source-selection/recovery report.")
    parser.add_argument("--source-mapping", required=True)
    parser.add_argument("--participant", default="P18")
    parser.add_argument("--cae-identity-qc")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    mapping = pd.read_csv(args.source_mapping, dtype=str).fillna("")
    mapping = mapping[mapping["participant"].eq(args.participant)].copy()

    qc_ok = None
    if args.cae_identity_qc and Path(args.cae_identity_qc).exists():
        qc = pd.read_csv(args.cae_identity_qc, dtype=str).fillna("")
        qc_ok = set(
            qc[
                qc["participant"].eq(args.participant)
                & qc["qc_status"].isin(["PASS", "WARN"])
            ]["video_id"].astype(str)
        )

    rows = []
    for attempt_id, group in mapping.groupby("attempt_id", sort=False):
        base = group.iloc[0]
        record = {
            "attempt_id": attempt_id,
            "participant": base["participant"],
            "attempt_sequence": base["attempt_sequence"],
            "round": base["round"],
            "device": base["device"],
        }

        for hand in ("left", "right"):
            hg = group[group["hand"].eq(hand)]
            gopro = hg[hg["source"].eq(f"gopro_{hand}")]
            cae = hg[hg["source"].eq("cae_hand")]

            gopro_row = gopro.iloc[0] if len(gopro) else None
            cae_row = cae.iloc[0] if len(cae) else None

            gopro_available = bool(gopro_row is not None and usable(gopro_row))
            cae_available = bool(cae_row is not None and usable(cae_row))

            if cae_available and qc_ok is not None:
                cae_available = str(cae_row["selected_video_id"]) in qc_ok

            if gopro_available:
                chosen = f"gopro_{hand}"
                reason = "Primary GoPro observation available"
            elif cae_available:
                chosen = "cae_hand"
                reason = "GoPro missing/unusable; validated CAE-HAND recovery available"
            else:
                chosen = ""
                reason = "No usable source mapped"

            record[f"gopro_{hand}_available"] = gopro_available
            record[f"cae_{hand}_available"] = cae_available
            record[f"chosen_{hand}_source"] = chosen
            record[f"{hand}_reason"] = reason

        rows.append(record)

    report = pd.DataFrame(rows)
    out_dir = ensure_dir(Path(args.output_root) / "analysis")
    path = out_dir / f"{args.participant}_source_recovery_report.csv"
    report.to_csv(path, index=False)
    print(f"Wrote: {path}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
