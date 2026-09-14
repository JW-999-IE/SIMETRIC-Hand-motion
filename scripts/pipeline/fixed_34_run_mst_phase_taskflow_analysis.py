from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
from scipy.stats import fisher_exact


EXCLUDED = {"P7", "P8", "P14"}
EVENTS = [
    ("Needle In", "Needle In (ms)", 1),
    ("Wire In", "Wire In (ms)", 2),
    ("Needle Out", "Needle Out (ms)", 3),
    ("Catheter Over Needle", "Catheter Over Needle (ms)", 4),
    ("Wire Out", "Wire Out (ms)", 5),
]
HIGHER = {"P1", "P3", "P9", "P12", "P29"}
SOME = {"P2", "P4", "P5", "P6", "P10", "P11", "P16", "P24", "P28"}


def pnorm(v):
    m = re.search(r"(\d+)", str(v))
    return f"P{int(m.group(1))}" if m else str(v)


def experience(p):
    p = pnorm(p)
    if p in HIGHER:
        return "Higher"
    if p in SOME:
        return "Some"
    return "None"


def device_family(v):
    s = str(v).strip().upper()
    if s.startswith("DEVICE1"): return "MST"
    if s.startswith("DEVICE2"): return "ATG"
    if s.startswith("DEVICE3"): return "CON"
    return ""


def device_rep(v):
    m = re.search(r"-(\d+)\s*$", str(v))
    return int(m.group(1)) if m else np.nan


def parse_times(v):
    if pd.isna(v):
        return []
    if isinstance(v, (int, float, np.integer, np.floating)):
        if np.isfinite(float(v)):
            return [float(v)]
        return []
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none"}:
        return []
    vals = []
    for token in re.split(r"[/;,|]+", s):
        token = token.strip()
        try:
            vals.append(float(token))
        except Exception:
            pass
    return vals


def gee_log_duration(df, predictor):
    d = df.copy()
    d["duration_sec"] = pd.to_numeric(d["duration_sec"], errors="coerce")
    d = d[(d["duration_sec"] > 0)].dropna(
        subset=["duration_sec", "participant", "experience_stratum",
                "device_repetition_c", "chronological_attempt_sequence_c", predictor]
    )
    if d[predictor].nunique() < 2 or d["participant"].nunique() < 8:
        return None
    d["_y"] = np.log(d["duration_sec"])
    formula = (
        f'_y ~ {predictor}'
        ' + C(experience_stratum, Treatment(reference="None"))'
        ' + device_repetition_c + chronological_attempt_sequence_c'
    )
    model = smf.gee(
        formula=formula, groups="participant", data=d,
        family=sm.families.Gaussian(), cov_struct=Exchangeable()
    )
    try:
        r = model.fit(cov_type="bias_reduced")
        cov = "bias_reduced"
    except Exception:
        r = model.fit(cov_type="robust")
        cov = "robust_fallback"
    term = predictor
    if term not in r.params.index:
        hits = [x for x in r.params.index if predictor in str(x)]
        if not hits:
            return None
        term = hits[0]
    b = float(r.params[term])
    se = float(r.bse[term])
    return {
        "predictor": predictor,
        "rows": len(d),
        "participants": d["participant"].nunique(),
        "cov_type": cov,
        "duration_ratio": math.exp(b),
        "ci95_low": math.exp(b - 1.95996398454 * se),
        "ci95_high": math.exp(b + 1.95996398454 * se),
        "p": float(r.pvalues[term]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--insertion-workbook", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.insertion_workbook, sheet_name=0, dtype=object)
    df["participant"] = df["Id"].map(pnorm)
    df = df[~df["participant"].isin(EXCLUDED)].copy()
    df["attempt_sequence"] = df.groupby("participant", sort=False).cumcount() + 1
    df["attempt_id"] = df["participant"] + "_T" + df["attempt_sequence"].astype(int).astype(str).str.zfill(2)
    df["device"] = df["Device"].map(device_family)
    df["device_repetition"] = df["Device"].map(device_rep)
    df["experience_stratum"] = df["participant"].map(experience)
    df["chronological_attempt_sequence"] = df["attempt_sequence"].astype(float)
    df["device_repetition_c"] = df["device_repetition"] - df["device_repetition"].mean()
    df["chronological_attempt_sequence_c"] = (
        df["chronological_attempt_sequence"] - df["chronological_attempt_sequence"].mean()
    )
    df["success"] = df["FTIS"].astype(str).str.strip().str.upper().eq("Y")
    df["duration_sec"] = pd.to_numeric(df["Duration (ms)"], errors="coerce") / 1000.0

    # Overall duration descriptive table for all LPC configurations.
    dur_summary = (
        df.groupby("device")["duration_sec"]
        .agg(["count", "median", "mean", "std"])
        .reset_index()
    )
    q = df.groupby("device")["duration_sec"].quantile([.25, .75]).unstack()
    q.columns = ["q25", "q75"]
    dur_summary = dur_summary.merge(q.reset_index(), on="device", how="left")
    dur_summary.to_csv(out / "lpc_total_duration_by_configuration.csv", index=False)

    mst = df[df["device"].eq("MST")].copy()
    attempts = []
    stream_rows = []
    phase_rows = []

    for _, r in mst.iterrows():
        start = pd.to_numeric(pd.Series([r["Start time (ms)"]]), errors="coerce").iloc[0]
        end = pd.to_numeric(pd.Series([r["End Time (ms)"]]), errors="coerce").iloc[0]

        lists = {}
        chronological = []
        for label, col, rank in EVENTS:
            vals = parse_times(r[col])
            lists[label] = vals
            for occ, ts in enumerate(vals, 1):
                chronological.append((ts, rank, label, occ))
                stream_rows.append({
                    "attempt_id": r["attempt_id"],
                    "participant": r["participant"],
                    "event": label,
                    "event_rank": rank,
                    "occurrence": occ,
                    "timestamp_ms": ts,
                    "time_from_attempt_start_sec": (ts - start) / 1000 if np.isfinite(start) else np.nan,
                })

        chronological.sort(key=lambda z: (z[0], z[1]))
        ranks = [x[1] for x in chronological]
        noncanonical = any(b < a for a, b in zip(ranks[:-1], ranks[1:]))

        counts = {label: len(lists[label]) for label, _, _ in EVENTS}
        all_present = all(counts[label] >= 1 for label, _, _ in EVENTS)
        exactly_one = all(counts[label] == 1 for label, _, _ in EVENTS)

        firsts = {label: min(lists[label]) if lists[label] else np.nan for label, _, _ in EVENTS}
        first_seq = [firsts[label] for label, _, _ in EVENTS]
        first_canonical = all(
            np.isfinite(a) and np.isfinite(b) and b > a
            for a, b in zip(first_seq[:-1], first_seq[1:])
        ) if all_present else False

        internal_missing = False
        for label, _, rank in EVENTS:
            if counts[label] == 0 and any(counts[l2] > 0 for l2, _, r2 in EVENTS if r2 > rank):
                internal_missing = True

        furthest = max([rank for label, _, rank in EVENTS if counts[label] > 0], default=0)
        terminal_incomplete = not all_present and not internal_missing
        repeated_total = sum(max(c - 1, 0) for c in counts.values())
        repeat_any = repeated_total > 0
        reentry = repeat_any or noncanonical

        attempts.append({
            "attempt_id": r["attempt_id"],
            "participant": r["participant"],
            "attempt_sequence": r["attempt_sequence"],
            "device_repetition": r["device_repetition"],
            "experience_stratum": r["experience_stratum"],
            "success": r["success"],
            "duration_sec": r["duration_sec"],
            "needle_in_count": counts["Needle In"],
            "wire_in_count": counts["Wire In"],
            "needle_out_count": counts["Needle Out"],
            "catheter_over_needle_count": counts["Catheter Over Needle"],
            "wire_out_count": counts["Wire Out"],
            "total_milestone_events": sum(counts.values()),
            "repeat_event_any": repeat_any,
            "repeated_event_count": repeated_total,
            "noncanonical_sequence": noncanonical,
            "all_five_milestones_observed": all_present,
            "exactly_one_each_milestone": exactly_one,
            "first_occurrences_canonical": first_canonical,
            "simple_complete_sequence": bool(all_present and exactly_one and first_canonical and not noncanonical),
            "furthest_milestone_rank": furthest,
            "internal_missing_milestone": internal_missing,
            "terminal_incomplete_sequence": terminal_incomplete,
            "procedural_reentry_candidate": reentry,
            "successful_after_reentry_candidate": bool(r["success"] and reentry),
        })

        # Primary phase timing: simple complete MST sequences only.
        if all_present and exactly_one and first_canonical and not noncanonical and np.isfinite(start) and np.isfinite(end):
            points = [
                ("Attempt start", start),
                ("Needle In", firsts["Needle In"]),
                ("Wire In", firsts["Wire In"]),
                ("Needle Out", firsts["Needle Out"]),
                ("Catheter Over Needle", firsts["Catheter Over Needle"]),
                ("Wire Out", firsts["Wire Out"]),
                ("Attempt end", end),
            ]
            for (a_name, a), (b_name, b) in zip(points[:-1], points[1:]):
                dur = (b - a) / 1000.0
                phase_rows.append({
                    "attempt_id": r["attempt_id"],
                    "participant": r["participant"],
                    "experience_stratum": r["experience_stratum"],
                    "device_repetition": r["device_repetition"],
                    "attempt_sequence": r["attempt_sequence"],
                    "phase": f"{a_name} -> {b_name}",
                    "phase_duration_sec": dur,
                    "total_attempt_duration_sec": r["duration_sec"],
                    "phase_fraction_of_total": dur / r["duration_sec"] if r["duration_sec"] and r["duration_sec"] > 0 else np.nan,
                })

    att = pd.DataFrame(attempts)
    stream = pd.DataFrame(stream_rows).sort_values(["participant", "attempt_id", "timestamp_ms"])
    phases = pd.DataFrame(phase_rows)

    # Add centered covariates for MST-only models.
    att["device_repetition_c"] = att["device_repetition"] - att["device_repetition"].mean()
    att["chronological_attempt_sequence_c"] = att["attempt_sequence"] - att["attempt_sequence"].mean()

    # Phase summaries.
    if len(phases):
        phase_summary = (
            phases.groupby("phase")
            .agg(
                attempts=("attempt_id", "nunique"),
                median_duration_sec=("phase_duration_sec", "median"),
                mean_duration_sec=("phase_duration_sec", "mean"),
                median_fraction_total=("phase_fraction_of_total", "median"),
                mean_fraction_total=("phase_fraction_of_total", "mean"),
            )
            .reset_index()
        )
        quant = phases.groupby("phase")["phase_duration_sec"].quantile([.25, .75]).unstack()
        quant.columns = ["q25_duration_sec", "q75_duration_sec"]
        phase_summary = phase_summary.merge(quant.reset_index(), on="phase", how="left")
    else:
        phase_summary = pd.DataFrame()

    # Flow-success exact associations.
    assoc_rows = []
    for flag in [
        "repeat_event_any",
        "noncanonical_sequence",
        "terminal_incomplete_sequence",
        "internal_missing_milestone",
        "procedural_reentry_candidate",
    ]:
        tab = pd.crosstab(att[flag], att["success"]).reindex(
            index=[False, True], columns=[False, True], fill_value=0
        )
        odds, p = fisher_exact(tab.to_numpy())
        assoc_rows.append({
            "flow_flag": flag,
            "n_flagged": int(att[flag].sum()),
            "success_flagged": int(att.loc[att[flag], "success"].sum()),
            "success_unflagged": int(att.loc[~att[flag], "success"].sum()),
            "fisher_odds_ratio": odds,
            "fisher_p": p,
        })
    assoc = pd.DataFrame(assoc_rows)

    # Duration association with repeat/re-entry patterns.
    duration_models = []
    for flag in ["repeat_event_any", "noncanonical_sequence", "procedural_reentry_candidate"]:
        x = att.copy()
        x[flag] = x[flag].astype(int)
        res = gee_log_duration(x, flag)
        if res:
            duration_models.append(res)
    duration_models = pd.DataFrame(duration_models)

    att.to_csv(out / "mst_taskflow_attempts.csv", index=False)
    stream.to_csv(out / "mst_event_stream.csv", index=False)
    phases.to_csv(out / "mst_simple_complete_phase_durations.csv", index=False)
    phase_summary.to_csv(out / "mst_phase_timing_summary.csv", index=False)
    assoc.to_csv(out / "mst_flow_success_associations.csv", index=False)
    duration_models.to_csv(out / "mst_flow_duration_models.csv", index=False)

    print("=== LPC MST PHASE / TASK-FLOW ANALYSIS COMPLETE ===")
    print("Included MST attempts:", len(att))
    print("Simple complete sequences:", int(att["simple_complete_sequence"].sum()))
    print("Any repeated milestone:", int(att["repeat_event_any"].sum()))
    print("Noncanonical sequence:", int(att["noncanonical_sequence"].sum()))
    print("Terminal incomplete sequence:", int(att["terminal_incomplete_sequence"].sum()))
    print("Internal missing milestone:", int(att["internal_missing_milestone"].sum()))
    print("Successful after re-entry candidate:", int(att["successful_after_reentry_candidate"].sum()))
    print()
    print("MST phase timing summary (simple complete sequences only):")
    print(phase_summary.to_string(index=False) if len(phase_summary) else "<none>")
    print()
    print("Flow-success exact associations:")
    print(assoc.to_string(index=False))
    print()
    print("Flow-duration GEE models:")
    print(duration_models.to_string(index=False) if len(duration_models) else "<none>")

if __name__ == "__main__":
    main()
