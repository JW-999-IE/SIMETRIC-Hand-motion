from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


HIGHER = {"P01", "P03", "P09", "P12", "P29"}
SOME = {"P02", "P04", "P05", "P06", "P10", "P11", "P16", "P24", "P28"}
NONE = {
    "P13", "P15", "P17", "P18", "P19", "P20", "P21",
    "P22", "P23", "P25", "P26", "P27", "P30"
}
EXPECTED = HIGHER | SOME | NONE
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def participant_norm(v):
    m = re.search(r"(\d+)", str(v))
    return f"P{int(m.group(1)):02d}" if m else str(v)


def experience(p):
    p = participant_norm(p)
    if p in HIGHER:
        return "Higher"
    if p in SOME:
        return "Some"
    if p in NONE:
        return "None"
    return ""


def parse_detail_repetition(v):
    m = re.match(r"^\s*DEVICE\d+-(\d+)\s*$", str(v), flags=re.I)
    return int(m.group(1)) if m else np.nan


def bool_str(v):
    return str(v).strip().lower() in {"true", "1", "yes"}


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Prepare SIMETRIC hand-motion modelling covariates and compare GoPro/all-source "
            "tracking-quality thresholds without using outcome significance. Adds the exact "
            "three-level prior-LPC-experience classification and creates threshold-specific "
            "analysis-set flags."
        )
    )
    ap.add_argument("--master", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    master = pd.read_csv(args.master, dtype=str).fillna("")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    master["participant"] = master["participant"].map(participant_norm)
    observed = set(master["participant"].unique())
    unknown = sorted(observed - EXPECTED)
    missing_expected = sorted(EXPECTED - observed)
    if unknown or missing_expected:
        print("STOP: participant-experience map does not reconcile.")
        print("Unknown participants:", unknown)
        print("Expected participants absent:", missing_expected)
        raise SystemExit(2)

    master["experience_stratum"] = master["participant"].map(experience)
    master["experience_any"] = master["experience_stratum"].map(
        {"None": 0, "Some": 1, "Higher": 1}
    )
    master["experience_higher_vs_rest"] = master["experience_stratum"].map(
        {"None": 0, "Some": 0, "Higher": 1}
    )

    # Preserve both chronological order and within-device repetition.
    master["chronological_attempt_sequence"] = to_num(master["attempt_sequence"])
    master["device_repetition"] = master["workbook_device_detail"].map(
        parse_detail_repetition
    )

    # Use one row per attempt for centring so left/right duplication has no effect.
    attempts = master[
        [
            "attempt_id", "participant", "chronological_attempt_sequence",
            "device_repetition", "round", "device"
        ]
    ].drop_duplicates("attempt_id").copy()

    seq_mean = attempts["chronological_attempt_sequence"].mean()
    rep_mean = attempts["device_repetition"].mean()

    master["chronological_attempt_sequence_c"] = (
        master["chronological_attempt_sequence"] - seq_mean
    )
    master["device_repetition_c"] = master["device_repetition"] - rep_mean

    # Round is kept independently. If numeric, expose a numeric version;
    # do not silently equate it with repetition.
    master["round_numeric"] = to_num(master["round"])

    qc_pass = master["qc_status"].eq("PASS")
    cov = to_num(master["tracking_coverage"])
    span = to_num(master["observed_span_fraction"])
    gopro = master["motion_source"].astype(str).str.startswith("gopro_")

    for thr in THRESHOLDS:
        tag = int(round(thr * 100))
        quality = qc_pass & cov.ge(thr) & span.ge(thr)
        master[f"quality_ge_{tag}"] = quality
        master[f"analysis_gopro_ge_{tag}"] = quality & gopro
        master[f"analysis_allsource_ge_{tag}"] = quality

    # Existing v3 flags retained for reconciliation.
    if "analysis_set_primary_gopro" in master.columns:
        master["existing_primary_gopro_bool"] = master[
            "analysis_set_primary_gopro"
        ].map(bool_str)
    if "analysis_set_all_sources_sensitivity" in master.columns:
        master["existing_allsource_sensitivity_bool"] = master[
            "analysis_set_all_sources_sensitivity"
        ].map(bool_str)

    # Threshold representation table.
    summary_rows = []
    for source_set in ["gopro", "allsource"]:
        for thr in THRESHOLDS:
            tag = int(round(thr * 100))
            flag = (
                master[f"analysis_gopro_ge_{tag}"]
                if source_set == "gopro"
                else master[f"analysis_allsource_ge_{tag}"]
            )
            sub = master[flag]
            groups = sub[["participant", "hand"]].drop_duplicates()
            summary_rows.append({
                "source_set": source_set,
                "threshold": thr,
                "rows": len(sub),
                "participants": sub["participant"].nunique(),
                "participant_hand_groups": len(groups),
                "left_rows": int((sub["hand"] == "left").sum()),
                "right_rows": int((sub["hand"] == "right").sum()),
                "left_participants": sub.loc[
                    sub["hand"].eq("left"), "participant"
                ].nunique(),
                "right_participants": sub.loc[
                    sub["hand"].eq("right"), "participant"
                ].nunique(),
                "MST_rows": int((sub["device"] == "MST").sum()),
                "ATG_rows": int((sub["device"] == "ATG").sum()),
                "CON_rows": int((sub["device"] == "CON").sum()),
            })

    summary = pd.DataFrame(summary_rows)
    summary_path = outdir / "tracking_threshold_representation.csv"
    summary.to_csv(summary_path, index=False)

    # Participant-hand rows available by threshold.
    ph_rows = []
    all_ph = master[["participant", "hand"]].drop_duplicates()
    for _, ph in all_ph.iterrows():
        p, h = ph["participant"], ph["hand"]
        g = master[
            master["participant"].eq(p) & master["hand"].eq(h)
        ]
        row = {
            "participant": p,
            "hand": h,
            "experience_stratum": experience(p),
            "attempts_expected": len(g),
            "mapping_unavailable": int(g["mapping_status"].eq("unavailable").sum()),
        }
        for thr in THRESHOLDS:
            tag = int(round(thr * 100))
            row[f"gopro_ge_{tag}_rows"] = int(
                g[f"analysis_gopro_ge_{tag}"].sum()
            )
            row[f"allsource_ge_{tag}_rows"] = int(
                g[f"analysis_allsource_ge_{tag}"].sum()
            )
        ph_rows.append(row)

    ph = pd.DataFrame(ph_rows)
    ph_path = outdir / "tracking_threshold_participant_hand.csv"
    ph.to_csv(ph_path, index=False)

    # Experience balance by threshold.
    exp_rows = []
    for source_set in ["gopro", "allsource"]:
        for thr in THRESHOLDS:
            tag = int(round(thr * 100))
            flag = (
                master[f"analysis_gopro_ge_{tag}"]
                if source_set == "gopro"
                else master[f"analysis_allsource_ge_{tag}"]
            )
            sub = master[flag]
            for exp, g in sub.groupby("experience_stratum"):
                exp_rows.append({
                    "source_set": source_set,
                    "threshold": thr,
                    "experience_stratum": exp,
                    "rows": len(g),
                    "participants": g["participant"].nunique(),
                    "participant_hand_groups": len(
                        g[["participant", "hand"]].drop_duplicates()
                    ),
                })
    exp_summary = pd.DataFrame(exp_rows)
    exp_path = outdir / "tracking_threshold_experience_balance.csv"
    exp_summary.to_csv(exp_path, index=False)

    # Round/repetition diagnostic.
    rep_diag = attempts.copy()
    rep_diag["round_numeric"] = to_num(rep_diag["round"])
    rep_diag["round_equals_device_repetition"] = (
        rep_diag["round_numeric"].notna()
        & rep_diag["device_repetition"].notna()
        & rep_diag["round_numeric"].eq(rep_diag["device_repetition"])
    )
    rep_path = outdir / "temporal_covariate_diagnostic.csv"
    rep_diag.to_csv(rep_path, index=False)

    output = outdir / "statistical_master_attempt_hand_828_v3_covariates.csv"
    master.to_csv(output, index=False)

    print("=== HAND-MOTION ANALYSIS POPULATION PREPARATION ===")
    print("Rows:", len(master))
    print("Participants:", master["participant"].nunique())
    print()
    print("Experience strata:")
    print(
        master[["participant", "experience_stratum"]]
        .drop_duplicates()
        ["experience_stratum"]
        .value_counts()
        .reindex(["Higher", "Some", "None"])
        .fillna(0)
        .astype(int)
        .to_string()
    )
    print()
    print("=== TRACKING-THRESHOLD REPRESENTATION ===")
    print(summary.to_string(index=False))
    print()

    numeric_round = attempts["round"].map(
        lambda x: pd.to_numeric(x, errors="coerce")
    )
    valid_round = numeric_round.notna() & attempts["device_repetition"].notna()
    if valid_round.any():
        agreement = (
            numeric_round[valid_round].astype(float).to_numpy()
            == attempts.loc[valid_round, "device_repetition"].astype(float).to_numpy()
        ).mean()
        print(f"Round vs device-repetition numeric agreement: {agreement:.1%}")
    else:
        print("Round vs device-repetition numeric agreement: not evaluable")

    print()
    print("Wrote:", output)
    print("Wrote:", summary_path)
    print("Wrote:", ph_path)
    print("Wrote:", exp_path)
    print("Wrote:", rep_path)
    print()
    print(
        "IMPORTANT: choose the modelling threshold from measurement quality and "
        "sample representation before inspecting device-effect model p-values."
    )


if __name__ == "__main__":
    main()
