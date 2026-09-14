from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


EXCLUDED = {"P7", "P8", "P14"}


def pnorm(v):
    m = re.search(r"(\d+)", str(v))
    return f"P{int(m.group(1)):02d}" if m else str(v)


def psimple(v):
    m = re.search(r"(\d+)", str(v))
    return f"P{int(m.group(1))}" if m else str(v)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def participant_summary(metrics: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    rows = []
    participants = sorted(
        [p for p in mapping["participant"].map(pnorm).unique() if psimple(p) not in EXCLUDED],
        key=lambda x: int(re.search(r"\d+", x).group())
    )

    for p in participants:
        for hand in ("left","right"):
            expected = mapping[
                mapping["participant"].map(pnorm).eq(p)
                & mapping["hand"].astype(str).str.lower().eq(hand)
            ]
            g = metrics[
                metrics["participant"].map(pnorm).eq(p)
                & metrics["hand"].astype(str).str.lower().eq(hand)
            ].copy()

            def mean(col):
                if col not in g.columns or g.empty:
                    return np.nan
                return pd.to_numeric(g[col], errors="coerce").mean()

            dominant = ""
            if "dominant_grip_candidate" in g.columns:
                s = g["dominant_grip_candidate"].dropna().astype(str)
                s = s[s.ne("")]
                if len(s):
                    dominant = s.mode().iloc[0]

            rows.append({
                "participant": p,
                "hand": hand,
                "included": True,
                "attempts_expected": len(expected),
                "attempts_with_metrics": len(g),
                "gopro_attempts": int(g.get("motion_source", pd.Series(dtype=str)).astype(str).str.startswith("gopro_").sum()) if len(g) else 0,
                "cae_recovered_attempts": int(g.get("motion_source", pd.Series(dtype=str)).astype(str).eq("cae_hand").sum()) if len(g) else 0,
                "missing_or_unusable_attempts": max(len(expected)-len(g),0),
                "mean_attempt_duration_sec": mean("attempt_duration_sec"),
                "mean_peak_accel_norm_s2": mean("peak_accel_norm_s2"),
                "mean_rms_accel_norm_s2": mean("rms_accel_norm_s2"),
                "mean_sparc": mean("sparc"),
                "mean_log_dimensionless_jerk": mean("log_dimensionless_jerk"),
                "mean_velocity_peak_count": mean("velocity_peak_count"),
                "mean_stillness_fraction": mean("stillness_fraction"),
                "mean_tremor_rms_norm": mean("tremor_rms_norm"),
                "mean_movement_onset_delay_sec": mean("movement_onset_delay_sec"),
                "dominant_grip_candidate": dominant,
                "grip_validation_status": "screening_only_manual_validation_required",
            })

    return pd.DataFrame(rows)


def write_excel(output, mapping, metrics, trajectory, grip):
    part = participant_summary(metrics, mapping)

    readme = pd.DataFrame([
        ["Study population", "P01-P30 excluding P07, P08 and P14."],
        ["Attempt-level unit", "One row per attempt × anatomical hand in Attempt_Hand_Summary where usable motion metrics exist."],
        ["Primary source", "GoPro-left / GoPro-right."],
        ["Secondary source", "CAE-HAND only where validated source hierarchy requires recovery."],
        ["Acceleration", "Mean, RMS and peak acceleration from the source-aware trajectory."],
        ["Smoothness", "SPARC plus log dimensionless jerk and velocity-peak count."],
        ["XY trajectory", "100-point normalized trajectory in Trajectory_100pt."],
        ["Stillness", "Total/fraction/episodes/longest stillness period."],
        ["Tremor", "Exploratory residual motion metric; tracking jitter may contribute."],
        ["Movement onset", "Delay from validated workbook-attempt start to first sustained directed motion."],
        ["Grip", "Candidate grip posture only until manual validation is completed."],
        ["Raw landmarks", "Remain in CSV/Parquet; not duplicated into Excel because cohort raw rows exceed Excel sheet limits."],
    ], columns=["Item","Definition"])

    qc_cols = [c for c in [
        "attempt_id","participant","hand","motion_source","video_id",
        "mapping_status","coverage","tracking_coverage","qc_status","review_note"
    ] if c in metrics.columns]
    qc = metrics[qc_cols].copy() if qc_cols else pd.DataFrame()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        part.to_excel(writer, sheet_name="Participant_Summary", index=False)
        metrics.to_excel(writer, sheet_name="Attempt_Hand_Summary", index=False)
        trajectory.to_excel(writer, sheet_name="Trajectory_100pt", index=False)
        grip.to_excel(writer, sheet_name="Grip_Segments", index=False)
        qc.to_excel(writer, sheet_name="QC_Source", index=False)

        book = writer.book
        header = book.add_format({
            "bold": True, "font_color": "white", "bg_color": "#1F4E78",
            "border": 1, "text_wrap": True, "valign": "top"
        })
        wrap = book.add_format({"text_wrap": True, "valign": "top"})

        for name, df in [
            ("README",readme),
            ("Participant_Summary",part),
            ("Attempt_Hand_Summary",metrics),
            ("Trajectory_100pt",trajectory),
            ("Grip_Segments",grip),
            ("QC_Source",qc),
        ]:
            ws=writer.sheets[name]
            ws.freeze_panes(1,0)
            if len(df.columns):
                ws.autofilter(0,0,max(len(df),1),len(df.columns)-1)
            for i,c in enumerate(df.columns):
                ws.write(0,i,c,header)
                width=min(max(len(str(c))+2,12),32)
                ws.set_column(i,i,width,wrap)

    print("Wrote:",output)
    print("Participant-hand rows:",len(part))
    print("Attempt-hand metric rows:",len(metrics))
    print("Trajectory rows:",len(trajectory))
    print("Grip segment rows:",len(grip))


def main():
    ap=argparse.ArgumentParser(description="Build final SIMETRIC cohort workbook directly from source-aware metric outputs.")
    ap.add_argument("--mapping",required=True)
    ap.add_argument("--source-aware-metrics",required=True)
    ap.add_argument("--trajectory",required=True)
    ap.add_argument("--grip-segments",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    mapping=read_csv(Path(args.mapping))
    metrics=read_csv(Path(args.source_aware_metrics))
    trajectory=read_csv(Path(args.trajectory))
    grip=read_csv(Path(args.grip_segments))

    if mapping.empty:
        raise RuntimeError("Frozen mapping is empty or missing.")
    if metrics.empty:
        raise RuntimeError("Source-aware metrics are empty or missing.")

    mapping["participant"]=mapping["participant"].map(pnorm)
    metrics["participant"]=metrics["participant"].map(pnorm)
    mapping=mapping[~mapping["participant"].map(psimple).isin(EXCLUDED)]
    metrics=metrics[~metrics["participant"].map(psimple).isin(EXCLUDED)]

    if not trajectory.empty and "participant" in trajectory.columns:
        trajectory["participant"]=trajectory["participant"].map(pnorm)
        trajectory=trajectory[~trajectory["participant"].map(psimple).isin(EXCLUDED)]
    if not grip.empty and "participant" in grip.columns:
        grip["participant"]=grip["participant"].map(pnorm)
        grip=grip[~grip["participant"].map(psimple).isin(EXCLUDED)]

    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)

    metrics.to_csv(out.parent/"attempt_hand_summary.csv",index=False)
    trajectory.to_csv(out.parent/"trajectory_100pt.csv",index=False)
    grip.to_csv(out.parent/"grip_segments_candidates.csv",index=False)
    participant_summary(metrics,mapping).to_csv(out.parent/"participant_hand_summary.csv",index=False)

    write_excel(out,mapping,metrics,trajectory,grip)

if __name__=="__main__":
    main()
