from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


EXCLUDED = {"P7", "P8", "P14"}

ATTEMPT_COLUMNS = [
    "attempt_id","participant","attempt_sequence","round","device","success","hand",
    "motion_source","video_id","mapping_status","coverage","local_start_sec","local_end_sec",
    "attempt_duration_sec","tracking_coverage",
    "start_x_norm","start_y_norm","end_x_norm","end_y_norm",
    "displacement_norm","path_length_norm","path_efficiency",
    "mean_speed_norm_s","peak_speed_norm_s",
    "mean_accel_norm_s2","rms_accel_norm_s2","peak_accel_norm_s2",
    "sparc","log_dimensionless_jerk","velocity_peak_count",
    "stillness_total_sec","stillness_fraction","stillness_episode_count","longest_stillness_sec",
    "tremor_rms_norm","tremor_dominant_hz",
    "movement_onset_delay_sec",
    "initial_grip_candidate","dominant_grip_candidate","final_grip_candidate",
    "grip_change_count","grip_validation_status",
    "qc_status","review_note",
]

ALIASES = {
    "attempt_id": ["attempt_id", "Attempt_ID"],
    "participant": ["participant", "Participant"],
    "hand": ["hand", "anatomical_hand", "target_hand"],
    "video_id": ["video_id", "VideoId"],
    "attempt_duration_sec": ["attempt_duration_sec", "duration_sec"],
    "tracking_coverage": ["tracking_coverage", "coverage", "target_coverage"],
    "start_x_norm": ["start_x_norm", "start_x", "x_start"],
    "start_y_norm": ["start_y_norm", "start_y", "y_start"],
    "end_x_norm": ["end_x_norm", "end_x", "x_end"],
    "end_y_norm": ["end_y_norm", "end_y", "y_end"],
    "displacement_norm": ["displacement_norm", "straight_line_displacement_2d", "displacement_2d"],
    "path_length_norm": ["path_length_norm", "path_length_diag", "path_length_2d", "path_length"],
    "path_efficiency": ["path_efficiency", "path_efficiency_2d"],
    "mean_speed_norm_s": ["mean_speed_norm_s", "mean_speed", "mean_velocity"],
    "peak_speed_norm_s": ["peak_speed_norm_s", "peak_speed", "peak_velocity"],
    "mean_accel_norm_s2": ["mean_accel_norm_s2", "mean_acceleration", "mean_accel"],
    "rms_accel_norm_s2": ["rms_accel_norm_s2", "rms_acceleration", "rms_accel"],
    "peak_accel_norm_s2": ["peak_accel_norm_s2", "peak_acceleration", "peak_accel"],
    "sparc": ["sparc", "SPARC"],
    "log_dimensionless_jerk": [
        "log_dimensionless_jerk",
        "log_dimensionless_jerk_smoothness",
        "ldlj",
    ],
    "velocity_peak_count": ["velocity_peak_count", "number_velocity_peaks", "nvp"],
    "stillness_total_sec": ["stillness_total_sec", "stillness_duration_sec"],
    "stillness_fraction": ["stillness_fraction", "idle_fraction"],
    "stillness_episode_count": ["stillness_episode_count", "stillness_episodes"],
    "longest_stillness_sec": ["longest_stillness_sec", "max_stillness_sec"],
    "tremor_rms_norm": ["tremor_rms_norm", "tremor_rms"],
    "tremor_dominant_hz": ["tremor_dominant_hz", "tremor_frequency_hz"],
    "movement_onset_delay_sec": ["movement_onset_delay_sec", "movement_onset_sec"],
    "movement_onset_delay_ms": ["movement_onset_delay_ms"],
    "initial_grip_candidate": ["initial_grip_candidate", "initial_grip"],
    "dominant_grip_candidate": ["dominant_grip_candidate", "dominant_grip"],
    "final_grip_candidate": ["final_grip_candidate", "final_grip"],
    "grip_change_count": ["grip_change_count", "posture_change_count"],
    "qc_status": ["qc_status", "status"],
}


def norm_participant(v: object) -> str:
    m = re.search(r"(\d+)", str(v))
    return f"P{int(m.group(1)):02d}" if m else str(v)


def simple_participant(v: object) -> str:
    m = re.search(r"(\d+)", str(v))
    return f"P{int(m.group(1))}" if m else str(v)


def norm_hand(v: object) -> str:
    s = str(v).strip().lower()
    if "left" in s:
        return "left"
    if "right" in s:
        return "right"
    return s


def find_col(df: pd.DataFrame, key: str):
    for name in ALIASES.get(key, [key]):
        if name in df.columns:
            return name
    return None


def get_series(df: pd.DataFrame, key: str, default=np.nan):
    col = find_col(df, key)
    if col is None:
        return pd.Series([default] * len(df), index=df.index)
    return df[col]


def standardize_metrics(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=df.index)
    for key in [
        "attempt_id","participant","hand","video_id",
        "attempt_duration_sec","tracking_coverage",
        "start_x_norm","start_y_norm","end_x_norm","end_y_norm",
        "displacement_norm","path_length_norm","path_efficiency",
        "mean_speed_norm_s","peak_speed_norm_s",
        "mean_accel_norm_s2","rms_accel_norm_s2","peak_accel_norm_s2",
        "sparc","log_dimensionless_jerk","velocity_peak_count",
        "stillness_total_sec","stillness_fraction","stillness_episode_count",
        "longest_stillness_sec","tremor_rms_norm","tremor_dominant_hz",
        "movement_onset_delay_sec","initial_grip_candidate",
        "dominant_grip_candidate","final_grip_candidate",
        "grip_change_count","qc_status",
    ]:
        out[key] = get_series(df, key)

    # Convert ms onset to sec if only ms is present.
    if out["movement_onset_delay_sec"].isna().all():
        ms_col = find_col(df, "movement_onset_delay_ms")
        if ms_col:
            out["movement_onset_delay_sec"] = pd.to_numeric(
                df[ms_col], errors="coerce"
            ) / 1000.0

    out["participant"] = out["participant"].map(norm_participant)
    out["hand"] = out["hand"].map(norm_hand)
    out["motion_source"] = source
    return out


def read_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_gopro_metrics(root: Path, explicit: Path | None) -> pd.DataFrame:
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates += [
        root / "metrics" / "attempt_hand_metrics.csv",
        root / "metrics" / "attempt_camera_metrics.csv",
    ]
    for path in candidates:
        if path and path.exists():
            df = pd.read_csv(path)
            out = standardize_metrics(df, "gopro")
            print(f"GoPro metrics: {path} ({len(out)} rows)")
            return out
    print("WARNING: no GoPro attempt-level metric file found.")
    return pd.DataFrame()


def load_cae_metrics(root: Path, explicit: Path | None) -> pd.DataFrame:
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates += [
        root / "cae" / "analysis" / "cae_motion.csv",
        root / "cae" / "analysis" / "motion_metrics.csv",
    ]
    for path in candidates:
        if path and path.exists():
            df = pd.read_csv(path)
            out = standardize_metrics(df, "cae_hand")
            print(f"CAE metrics: {path} ({len(out)} rows)")
            return out
    print("WARNING: no CAE motion metric file found.")
    return pd.DataFrame()


def source_match(metric_source: str, chosen_source: str) -> bool:
    chosen_source = str(chosen_source)
    metric_source = str(metric_source)
    if metric_source == chosen_source:
        return True
    if chosen_source.startswith("gopro_") and metric_source == "gopro":
        return True
    return False


def choose_metrics(mapping: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    records = []

    for _, m in mapping.iterrows():
        p = norm_participant(m["participant"])
        if simple_participant(p) in EXCLUDED:
            continue

        hand = norm_hand(m["hand"])
        chosen_source = str(m.get("chosen_source", ""))
        attempt_id = str(m["attempt_id"])

        row = {
            "attempt_id": attempt_id,
            "participant": p,
            "attempt_sequence": m.get("attempt_sequence", ""),
            "round": m.get("round", ""),
            "device": m.get("device", ""),
            "success": m.get("success", ""),
            "hand": hand,
            "motion_source": chosen_source,
            "video_id": m.get("selected_video_id", ""),
            "mapping_status": m.get("mapping_status", ""),
            "coverage": m.get("coverage", ""),
            "local_start_sec": m.get("local_start_sec", ""),
            "local_end_sec": m.get("local_end_sec", ""),
            "grip_validation_status": "screening_only_manual_validation_required",
            "review_note": m.get("review_note", ""),
        }

        if metrics.empty or chosen_source == "":
            records.append(row)
            continue

        candidates = metrics[
            metrics["attempt_id"].astype(str).eq(attempt_id)
            & metrics["participant"].astype(str).eq(p)
            & metrics["hand"].astype(str).eq(hand)
        ].copy()

        if not candidates.empty:
            candidates = candidates[
                candidates["motion_source"].map(
                    lambda s: source_match(str(s), chosen_source)
                )
            ]

        if len(candidates):
            x = candidates.iloc[0]
            for c in ATTEMPT_COLUMNS:
                if c in row:
                    continue
                if c in x.index:
                    row[c] = x[c]

        records.append(row)

    out = pd.DataFrame(records)
    for c in ATTEMPT_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan

    # Derive duration where possible.
    start = pd.to_numeric(out["local_start_sec"], errors="coerce")
    end = pd.to_numeric(out["local_end_sec"], errors="coerce")
    dur = pd.to_numeric(out["attempt_duration_sec"], errors="coerce")
    out.loc[dur.isna(), "attempt_duration_sec"] = end - start

    return out[ATTEMPT_COLUMNS]


def resample_group(g: pd.DataFrame, n=100) -> pd.DataFrame:
    time_col = None
    for c in ["normalized_time_0_1","time_norm","relative_time","video_time_sec","timestamp_sec","time_sec"]:
        if c in g.columns:
            time_col = c
            break
    x_col = next((c for c in ["x_norm","wrist_x_norm","x","wrist_x"] if c in g.columns), None)
    y_col = next((c for c in ["y_norm","wrist_y_norm","y","wrist_y"] if c in g.columns), None)
    if not time_col or not x_col or not y_col:
        return pd.DataFrame()

    t = pd.to_numeric(g[time_col], errors="coerce")
    x = pd.to_numeric(g[x_col], errors="coerce")
    y = pd.to_numeric(g[y_col], errors="coerce")
    valid = t.notna() & x.notna() & y.notna()
    if valid.sum() < 3:
        return pd.DataFrame()

    t = t[valid].to_numpy(float)
    x = x[valid].to_numpy(float)
    y = y[valid].to_numpy(float)
    order = np.argsort(t)
    t, x, y = t[order], x[order], y[order]

    if time_col != "normalized_time_0_1":
        if t[-1] <= t[0]:
            return pd.DataFrame()
        t = (t - t[0]) / (t[-1] - t[0])

    target = np.linspace(0, 1, n)
    return pd.DataFrame({
        "trajectory_point": np.arange(1, n + 1),
        "normalized_time_0_1": target,
        "x_norm": np.interp(target, t, x),
        "y_norm": np.interp(target, t, y),
    })


def load_gopro_trajectories(root: Path, summary: pd.DataFrame) -> pd.DataFrame:
    folder = root / "metrics" / "trajectories"
    if not folder.exists():
        return pd.DataFrame()

    files = sorted(folder.glob("*.csv*"))
    rows = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        aid_col = next((c for c in ["attempt_id","Attempt_ID"] if c in df.columns), None)
        hand_col = next((c for c in ["hand","anatomical_hand","target_hand"] if c in df.columns), None)
        if not aid_col:
            continue

        if hand_col:
            groups = df.groupby([aid_col, hand_col], dropna=False)
        else:
            groups = [(("", ""), df)]

        for key, g in groups:
            if hand_col:
                aid, hand = key
            else:
                aid = str(g[aid_col].iloc[0])
                # Infer hand from summary only if unique.
                h = summary[
                    summary["attempt_id"].astype(str).eq(str(aid))
                    & summary["motion_source"].astype(str).str.startswith("gopro_")
                ]["hand"].dropna().unique()
                if len(h) != 1:
                    continue
                hand = h[0]

            r = resample_group(g, 100)
            if r.empty:
                continue
            match = summary[
                summary["attempt_id"].astype(str).eq(str(aid))
                & summary["hand"].eq(norm_hand(hand))
            ]
            if match.empty:
                continue
            m = match.iloc[0]
            r.insert(0, "video_id", m["video_id"])
            r.insert(0, "motion_source", m["motion_source"])
            r.insert(0, "hand", m["hand"])
            r.insert(0, "participant", m["participant"])
            r.insert(0, "attempt_id", m["attempt_id"])
            rows.append(r)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_cae_trajectories(root: Path, summary: pd.DataFrame) -> pd.DataFrame:
    reconstructed_root = root / "cae" / "analysis" / "reconstructed"
    if not reconstructed_root.exists():
        return pd.DataFrame()

    rows = []
    cae_rows = summary[
        summary["motion_source"].eq("cae_hand")
        & summary["mapping_status"].isin(
            ["validated","partial_start","partial_end","partial_both"]
        )
    ]

    for _, m in cae_rows.iterrows():
        video_id = str(m["video_id"])
        folder = reconstructed_root / video_id
        candidates = [
            folder / "reconstructed.parquet",
            folder / "reconstructed.csv.gz",
            folder / "reconstructed.csv",
        ]
        df = None
        for path in candidates:
            if path.exists():
                if path.suffix == ".parquet":
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_csv(path)
                break
        if df is None or df.empty:
            continue

        hand = m["hand"]
        if "anatomical_hand" in df.columns:
            df = df[df["anatomical_hand"].astype(str).str.lower().eq(hand)]

        if "landmark_index" in df.columns:
            df = df[pd.to_numeric(df["landmark_index"], errors="coerce").eq(0)]

        time_col = next((c for c in ["timestamp_sec","video_time_sec"] if c in df.columns), None)
        if not time_col:
            continue

        t = pd.to_numeric(df[time_col], errors="coerce")
        start = pd.to_numeric(pd.Series([m["local_start_sec"]]), errors="coerce").iloc[0]
        end = pd.to_numeric(pd.Series([m["local_end_sec"]]), errors="coerce").iloc[0]
        df = df[t.between(start, end)]
        if df.empty:
            continue

        r = resample_group(df, 100)
        if r.empty:
            continue
        r.insert(0, "video_id", video_id)
        r.insert(0, "motion_source", "cae_hand")
        r.insert(0, "hand", hand)
        r.insert(0, "participant", m["participant"])
        r.insert(0, "attempt_id", m["attempt_id"])
        rows.append(r)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def participant_summary(attempt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (participant, hand), g in attempt.groupby(["participant","hand"], dropna=False):
        usable = g[g["mapping_status"].isin(
            ["validated","partial_start","partial_end","partial_both"]
        )]
        numeric = lambda c: pd.to_numeric(usable[c], errors="coerce")

        dominant = usable["dominant_grip_candidate"].dropna().astype(str)
        dominant = dominant[dominant.ne("")]
        dominant_value = dominant.mode().iloc[0] if len(dominant) else ""

        rows.append({
            "participant": participant,
            "hand": hand,
            "included": True,
            "attempts_expected": len(g),
            "attempts_with_usable_motion": len(usable),
            "gopro_attempts": int(usable["motion_source"].astype(str).str.startswith("gopro_").sum()),
            "cae_recovered_attempts": int(usable["motion_source"].eq("cae_hand").sum()),
            "missing_attempts": len(g) - len(usable),
            "mean_attempt_duration_sec": numeric("attempt_duration_sec").mean(),
            "mean_peak_accel_norm_s2": numeric("peak_accel_norm_s2").mean(),
            "mean_rms_accel_norm_s2": numeric("rms_accel_norm_s2").mean(),
            "mean_sparc": numeric("sparc").mean(),
            "mean_log_dimensionless_jerk": numeric("log_dimensionless_jerk").mean(),
            "mean_velocity_peak_count": numeric("velocity_peak_count").mean(),
            "mean_stillness_fraction": numeric("stillness_fraction").mean(),
            "mean_tremor_rms_norm": numeric("tremor_rms_norm").mean(),
            "mean_movement_onset_delay_sec": numeric("movement_onset_delay_sec").mean(),
            "dominant_grip_candidate": dominant_value,
            "grip_validation_status": "screening_only_manual_validation_required",
            "qc_status": "",
        })

    return pd.DataFrame(rows).sort_values(["participant","hand"])


def write_excel(
    output: Path,
    participant: pd.DataFrame,
    attempt: pd.DataFrame,
    trajectory: pd.DataFrame,
):
    output.parent.mkdir(parents=True, exist_ok=True)

    readme = pd.DataFrame([
        ["Purpose", "One row per attempt × anatomical hand; P07/P08/P14 excluded."],
        ["Primary source", "GoPro-left / GoPro-right."],
        ["Secondary source", "CAE-HAND only where GoPro is missing/unusable."],
        ["Raw landmark data", "Keep in CSV/Parquet. Do not put all raw rows in Excel."],
        ["Acceleration", "Mean/RMS/peak acceleration from validated attempt-level trajectory."],
        ["Smoothness", "SPARC primary; log dimensionless jerk and velocity peaks secondary."],
        ["XY trajectory", "100 time-normalized points per attempt × hand for plotting/comparison."],
        ["Grip", "Candidate IPG/EPG/TC/NOT_GRIPPING/UNCERTAIN; manual validation required before inferential use."],
        ["Stillness", "Duration/fraction/episode count/longest episode."],
        ["Tremor", "Exploratory; tracking jitter may mimic tremor."],
        ["Movement onset", "Delay from validated attempt start to first sustained insertion movement."],
    ], columns=["Item","Definition"])

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        participant.to_excel(writer, sheet_name="Participant_Summary", index=False)
        attempt.to_excel(writer, sheet_name="Attempt_Hand_Summary", index=False)
        trajectory.to_excel(writer, sheet_name="Trajectory_100pt", index=False)

        workbook = writer.book
        header_fmt = workbook.add_format({
            "bold": True, "font_color": "white", "bg_color": "#1F4E78",
            "border": 1, "text_wrap": True, "valign": "top",
        })
        wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
        num_fmt = workbook.add_format({"num_format": "0.000"})

        for sheet_name, df in [
            ("README", readme),
            ("Participant_Summary", participant),
            ("Attempt_Hand_Summary", attempt),
            ("Trajectory_100pt", trajectory),
        ]:
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, max(len(df), 1), max(len(df.columns)-1, 0))
            for col_num, col_name in enumerate(df.columns):
                ws.write(0, col_num, col_name, header_fmt)
                width = min(max(len(str(col_name)) + 2, 12), 30)
                ws.set_column(col_num, col_num, width, wrap_fmt)

    print(f"Wrote workbook: {output}")


def main():
    ap = argparse.ArgumentParser(
        description="Build final SIMETRIC participant/attempt hand-motion summary workbook."
    )
    ap.add_argument("--root", required=True,
                    help="SIMETRIC output root")
    ap.add_argument("--mapping", required=True,
                    help="Frozen source-aware attempt map")
    ap.add_argument("--source-aware-metrics")
    ap.add_argument("--gopro-metrics")
    ap.add_argument("--cae-metrics")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    mapping = pd.read_csv(args.mapping, dtype=str).fillna("")
    mapping["participant"] = mapping["participant"].map(norm_participant)
    mapping["hand"] = mapping["hand"].map(norm_hand)
    mapping = mapping[
        ~mapping["participant"].map(simple_participant).isin(EXCLUDED)
    ].copy()

    if args.source_aware_metrics:
        raw = pd.read_csv(args.source_aware_metrics)
        # This file is already one row per mapped attempt × hand with provenance.
        metrics = raw.copy()
        metrics["participant"] = metrics["participant"].map(norm_participant)
        metrics["hand"] = metrics["hand"].map(norm_hand)
        if "motion_source" not in metrics.columns and "chosen_source" in metrics.columns:
            metrics["motion_source"] = metrics["chosen_source"]
        print(f"Source-aware metrics: {args.source_aware_metrics} ({len(metrics)} rows)")
    else:
        gopro = load_gopro_metrics(
            root,
            Path(args.gopro_metrics) if args.gopro_metrics else None
        )
        cae = load_cae_metrics(
            root,
            Path(args.cae_metrics) if args.cae_metrics else None
        )
        metrics = pd.concat([gopro, cae], ignore_index=True) if len(gopro) or len(cae) else pd.DataFrame()

    attempt = choose_metrics(mapping, metrics)
    participant = participant_summary(attempt)

    gtraj = load_gopro_trajectories(root, attempt)
    ctraj = load_cae_trajectories(root, attempt)
    trajectory = pd.concat([gtraj, ctraj], ignore_index=True) if len(gtraj) or len(ctraj) else pd.DataFrame(
        columns=["attempt_id","participant","hand","motion_source","video_id",
                 "trajectory_point","normalized_time_0_1","x_norm","y_norm"]
    )

    # Machine-readable CSV companions.
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    participant.to_csv(out_dir / "participant_hand_summary.csv", index=False)
    attempt.to_csv(out_dir / "attempt_hand_summary.csv", index=False)
    trajectory.to_csv(out_dir / "trajectory_100pt.csv", index=False)

    write_excel(Path(args.output), participant, attempt, trajectory)

    print()
    print("=== SUMMARY ===")
    print(f"Included participants: {attempt['participant'].nunique()}")
    print(f"Attempt-hand rows: {len(attempt)}")
    print(f"Usable mapped rows: {attempt['mapping_status'].isin(['validated','partial_start','partial_end','partial_both']).sum()}")
    print(f"Trajectory rows: {len(trajectory)}")


if __name__ == "__main__":
    main()
