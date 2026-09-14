from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Build a stratified manual-validation sample from grip candidate segments. "
            "This does not validate grip automatically; it creates the review manifest."
        )
    )
    ap.add_argument("--grip-segments", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--per-stratum", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260817)
    args = ap.parse_args()

    df = pd.read_csv(args.grip_segments)
    required = {
        "attempt_id","participant","hand","motion_source","video_id",
        "segment_number","segment_start_sec","segment_end_sec",
        "segment_duration_sec","grip_candidate","candidate_confidence"
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            "Grip segment file is missing: " + ", ".join(sorted(missing))
        )

    df = df.copy()
    df["segment_duration_sec"] = pd.to_numeric(
        df["segment_duration_sec"], errors="coerce"
    )
    df["candidate_confidence"] = pd.to_numeric(
        df["candidate_confidence"], errors="coerce"
    )
    df["review_midpoint_sec"] = (
        pd.to_numeric(df["segment_start_sec"], errors="coerce")
        + pd.to_numeric(df["segment_end_sec"], errors="coerce")
    ) / 2.0

    # Exclude vanishingly short segments from the validation sample.
    pool = df[df["segment_duration_sec"] >= 0.5].copy()

    strata = ["motion_source","hand","grip_candidate"]
    samples = []
    for key, g in pool.groupby(strata, dropna=False):
        n = min(args.per_stratum, len(g))
        if n:
            samples.append(g.sample(n=n, random_state=args.seed))

    out = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()

    if len(out):
        out = out.sort_values(
            ["motion_source","hand","grip_candidate","participant","attempt_id"]
        ).reset_index(drop=True)
        out.insert(0, "review_id", [f"G{i:04d}" for i in range(1, len(out)+1)])

    out["manual_label"] = ""
    out["manual_gripping_present"] = ""
    out["manual_segment_start_sec"] = ""
    out["manual_segment_end_sec"] = ""
    out["reviewer"] = ""
    out["review_status"] = "pending"
    out["review_note"] = ""

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    summary = (
        out.groupby(strata, dropna=False)
        .size().reset_index(name="sampled_segments")
        if len(out) else pd.DataFrame()
    )
    summary_path = output.with_name(output.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("=== GRIP VALIDATION SAMPLE CREATED ===")
    print("Sampled segments:", len(out))
    print("Wrote:", output)
    print("Wrote:", summary_path)
    print()
    print(
        "Grip remains screening-only until this sample (or a larger pre-specified sample) "
        "is manually reviewed and agreement/accuracy is quantified."
    )


if __name__ == "__main__":
    main()
