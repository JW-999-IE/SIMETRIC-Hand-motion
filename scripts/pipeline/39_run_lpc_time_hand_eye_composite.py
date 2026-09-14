from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2, norm
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.stats.multitest import multipletests


PAIRWISE = [
    ("MST_vs_CON", "MST", "CON"),
    ("ATG_vs_CON", "ATG", "CON"),
    ("MST_vs_ATG", "MST", "ATG"),
]


def as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def canonical_attempt_id(v) -> str:
    s = str(v).strip()
    m = re.search(r"P0*(\d+).*?T0*(\d+)", s, flags=re.I)
    if m:
        return f"P{int(m.group(1))}_T{int(m.group(2)):02d}"
    return s


def canonical_participant(v) -> str:
    s = str(v).strip()
    m = re.search(r"P0*(\d+)", s, flags=re.I)
    if m:
        return f"P{int(m.group(1))}"
    return s


def canonical_device(v) -> str:
    s = str(v).strip().upper()
    if "MST" in s or s.startswith("DEVICE1"):
        return "MST"
    if "ATG" in s or s.startswith("DEVICE2"):
        return "ATG"
    if "CON" in s or s.startswith("DEVICE3"):
        return "CON"
    return s


def zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean()) / sd


def pick_unique(df: pd.DataFrame, key: str, source_name: str) -> pd.DataFrame:
    dup = df[key].duplicated(keep=False)
    if dup.any():
        sample = df.loc[dup, [key]].drop_duplicates().head(20)
        raise RuntimeError(
            f"{source_name} has duplicate attempt rows after ID normalization. "
            f"Examples: {sample[key].tolist()}"
        )
    return df


def device_term(names, level):
    hits = [
        n for n in names
        if "C(device" in str(n) and f"[T.{level}]" in str(n)
    ]
    if len(hits) != 1:
        raise RuntimeError(f"Could not uniquely locate device term for {level}: {hits}")
    return hits[0]


def omnibus_device(result):
    names = list(result.params.index)
    idx = [names.index(device_term(names, x)) for x in ["ATG", "MST"]]
    b = result.params.to_numpy(float)[idx]
    cov = np.asarray(result.cov_params(), dtype=float)[np.ix_(idx, idx)]
    stat = float(b.T @ np.linalg.pinv(cov) @ b)
    return stat, 2, float(chi2.sf(stat, 2))


def pairwise_device(result):
    names = list(result.params.index)
    rows = []
    for label, high, low in PAIRWISE:
        c = np.zeros(len(names), dtype=float)
        if high != "CON":
            c[names.index(device_term(names, high))] += 1
        if low != "CON":
            c[names.index(device_term(names, low))] -= 1
        beta = float(c @ result.params.to_numpy(float))
        cov = np.asarray(result.cov_params(), dtype=float)
        se = math.sqrt(max(float(c @ cov @ c), 0.0))
        z = beta / se if se > 0 else np.nan
        p = 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
        rows.append({
            "contrast": label,
            "mean_difference_composite_sd_units": beta,
            "ci95_low": beta - 1.95996398454 * se,
            "ci95_high": beta + 1.95996398454 * se,
            "p_raw": p,
        })
    q = pd.DataFrame(rows)
    valid = q["p_raw"].notna()
    if valid.any():
        _, adj, _, _ = multipletests(q.loc[valid, "p_raw"], method="holm")
        q.loc[valid, "p_holm"] = adj
    return q


def fit_composite(d: pd.DataFrame, score_col: str, label: str):
    x = d.dropna(
        subset=[
            score_col, "participant", "device", "experience_stratum",
            "device_repetition_c", "chronological_attempt_sequence_c",
        ]
    ).copy()

    if x["device"].nunique() < 3:
        raise RuntimeError(f"{label}: fewer than 3 LPC configurations remain")
    if x["participant"].nunique() < 8:
        raise RuntimeError(f"{label}: insufficient participants")

    formula = (
        f'{score_col} ~ C(device, Treatment(reference="CON"))'
        ' + C(experience_stratum, Treatment(reference="None"))'
        ' + device_repetition_c'
        ' + chronological_attempt_sequence_c'
    )
    model = smf.gee(
        formula=formula,
        groups="participant",
        data=x,
        family=sm.families.Gaussian(),
        cov_struct=Exchangeable(),
    )
    try:
        result = model.fit(cov_type="bias_reduced", maxiter=200)
        cov_type = "bias_reduced"
    except Exception:
        result = model.fit(cov_type="robust", maxiter=200)
        cov_type = "robust_fallback"

    stat, df_test, p = omnibus_device(result)
    pairs = pairwise_device(result)
    return result, x, cov_type, stat, df_test, p, pairs


def build_score(df: pd.DataFrame, mode: str, require_both_hands: bool):
    """
    mode='rate': primary duration-decoupled activity score.
    mode='count': protocol-faithful count sensitivity.
    """
    d = df.copy()

    if mode == "rate":
        eye_a = "fixation_rate_per_second"
        eye_b = "saccade_rate_per_second"
        hand_left = "hand_velocity_peak_rate_left"
        hand_right = "hand_velocity_peak_rate_right"
    elif mode == "count":
        eye_a = "fixation_count"
        eye_b = "saccade_count"
        hand_left = "velocity_peak_count_left"
        hand_right = "velocity_peak_count_right"
    else:
        raise ValueError(mode)

    required = [
        "attempt_duration_sec", eye_a, eye_b,
    ]
    d = d.dropna(subset=required).copy()
    d = d[d["attempt_duration_sec"] > 0].copy()

    if require_both_hands:
        d = d.dropna(subset=[hand_left, hand_right]).copy()
    else:
        d = d[d[[hand_left, hand_right]].notna().any(axis=1)].copy()

    # Standardisation is global across the analysis population, not by device.
    d["time_component_z"] = zscore(np.log(d["attempt_duration_sec"]))

    d["eye_a_z"] = zscore(d[eye_a])
    d["eye_b_z"] = zscore(d[eye_b])
    d["eye_activity_z"] = d[["eye_a_z", "eye_b_z"]].mean(axis=1)

    d["hand_left_z"] = zscore(d[hand_left])
    d["hand_right_z"] = zscore(d[hand_right])
    d["hand_activity_z"] = d[["hand_left_z", "hand_right_z"]].mean(axis=1, skipna=True)

    d["time_hand_eye_composite"] = d[
        ["time_component_z", "hand_activity_z", "eye_activity_z"]
    ].mean(axis=1)

    d["composite_mode"] = mode
    d["require_both_hands"] = require_both_hands
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--fixation-file", required=True)
    ap.add_argument("--saccade-file", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Frozen hand master ---
    master = pd.read_csv(
        args.master,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    master["attempt_key"] = master["attempt_id"].map(canonical_attempt_id)
    master["participant"] = master["participant"].map(canonical_participant)
    master["device"] = master["device"].map(canonical_device)

    for c in [
        "device_repetition_c",
        "chronological_attempt_sequence_c",
        "velocity_peak_count",
        "observed_continuous_tracking_sec",
    ]:
        master[c] = pd.to_numeric(master[c], errors="coerce")

    # One attempt-level metadata row, with fail-closed consistency checks.
    meta_cols = [
        "attempt_key", "participant", "device", "experience_stratum",
        "device_repetition_c", "chronological_attempt_sequence_c",
    ]
    for col in meta_cols[1:]:
        nun = master.groupby("attempt_key")[col].nunique(dropna=False)
        bad = nun[nun > 1]
        if len(bad):
            raise RuntimeError(
                f"Frozen master has inconsistent attempt-level {col}: "
                f"{bad.head(10).to_dict()}"
            )
    meta = master[meta_cols].drop_duplicates("attempt_key")

    # Main hand eligibility from the frozen GoPro >=60% population.
    hand = master[as_bool(master["analysis_main"])].copy()
    hand = hand[hand["hand"].isin(["left", "right"])].copy()
    hand["hand_velocity_peak_rate"] = (
        hand["velocity_peak_count"] / hand["observed_continuous_tracking_sec"]
    )
    hand.loc[
        ~(hand["observed_continuous_tracking_sec"] > 0),
        "hand_velocity_peak_rate",
    ] = np.nan

    hand_keep = hand[
        [
            "attempt_key", "hand", "velocity_peak_count",
            "hand_velocity_peak_rate",
        ]
    ].copy()

    if hand_keep.duplicated(["attempt_key", "hand"]).any():
        raise RuntimeError("Frozen hand master is not unique by attempt_key × hand")

    hand_wide = hand_keep.pivot(
        index="attempt_key",
        columns="hand",
        values=["velocity_peak_count", "hand_velocity_peak_rate"],
    )
    hand_wide.columns = [
        f"{metric}_{hand_side}" for metric, hand_side in hand_wide.columns
    ]
    hand_wide = hand_wide.reset_index()

    # --- Eye fixation data ---
    fix = pd.read_csv(args.fixation_file)
    required_fix = {
        "Attempt_ID", "Participant", "Device_name", "Attempt_duration_ms",
        "Fixation_count", "Fixation_rate_per_second",
    }
    missing = required_fix - set(fix.columns)
    if missing:
        raise RuntimeError(f"Fixation file missing columns: {sorted(missing)}")

    fix = fix[list(required_fix)].copy()
    fix["attempt_key"] = fix["Attempt_ID"].map(canonical_attempt_id)
    fix["participant_eye_fix"] = fix["Participant"].map(canonical_participant)
    fix["device_eye_fix"] = fix["Device_name"].map(canonical_device)
    fix["attempt_duration_sec"] = pd.to_numeric(
        fix["Attempt_duration_ms"], errors="coerce"
    ) / 1000.0
    fix["fixation_count"] = pd.to_numeric(fix["Fixation_count"], errors="coerce")
    fix["fixation_rate_per_second"] = pd.to_numeric(
        fix["Fixation_rate_per_second"], errors="coerce"
    )
    fix = pick_unique(fix, "attempt_key", "fixation file")

    # --- Eye saccade data ---
    sac = pd.read_csv(args.saccade_file)
    required_sac = {
        "Attempt_ID", "Participant", "Device_name", "Attempt_duration_ms",
        "Saccade_count", "Saccade_rate_per_second",
    }
    missing = required_sac - set(sac.columns)
    if missing:
        raise RuntimeError(f"Saccade file missing columns: {sorted(missing)}")

    sac = sac[list(required_sac)].copy()
    sac["attempt_key"] = sac["Attempt_ID"].map(canonical_attempt_id)
    sac["participant_eye_sac"] = sac["Participant"].map(canonical_participant)
    sac["device_eye_sac"] = sac["Device_name"].map(canonical_device)
    sac["attempt_duration_sec_saccade"] = pd.to_numeric(
        sac["Attempt_duration_ms"], errors="coerce"
    ) / 1000.0
    sac["saccade_count"] = pd.to_numeric(sac["Saccade_count"], errors="coerce")
    sac["saccade_rate_per_second"] = pd.to_numeric(
        sac["Saccade_rate_per_second"], errors="coerce"
    )
    sac = pick_unique(sac, "attempt_key", "saccade file")

    eye = fix.merge(
        sac[
            [
                "attempt_key", "participant_eye_sac", "device_eye_sac",
                "attempt_duration_sec_saccade",
                "saccade_count", "saccade_rate_per_second",
            ]
        ],
        on="attempt_key",
        how="inner",
        validate="one_to_one",
    )

    # Fail closed on eye-source mismatches.
    eye_part_bad = eye["participant_eye_fix"] != eye["participant_eye_sac"]
    eye_dev_bad = eye["device_eye_fix"] != eye["device_eye_sac"]
    duration_delta = (
        eye["attempt_duration_sec"] - eye["attempt_duration_sec_saccade"]
    ).abs()
    if eye_part_bad.any():
        raise RuntimeError("Fixation/saccade participant mismatch detected")
    if eye_dev_bad.any():
        raise RuntimeError("Fixation/saccade LPC-configuration mismatch detected")
    if duration_delta.dropna().gt(0.001).any():
        raise RuntimeError("Fixation/saccade attempt-duration mismatch >1 ms detected")

    # Merge eye + frozen attempt metadata + eligible hand data.
    merged = (
        meta.merge(eye, on="attempt_key", how="inner", validate="one_to_one")
        .merge(hand_wide, on="attempt_key", how="left", validate="one_to_one")
    )

    # Frozen master remains authoritative.
    participant_mismatch = merged["participant"] != merged["participant_eye_fix"]
    device_mismatch = merged["device"] != merged["device_eye_fix"]
    if participant_mismatch.any():
        bad = merged.loc[participant_mismatch, ["attempt_key", "participant", "participant_eye_fix"]]
        raise RuntimeError(
            "Frozen master vs eye participant mismatch: "
            + bad.head(10).to_dict("records").__repr__()
        )
    if device_mismatch.any():
        bad = merged.loc[device_mismatch, ["attempt_key", "device", "device_eye_fix"]]
        raise RuntimeError(
            "Frozen master vs eye LPC-configuration mismatch: "
            + bad.head(10).to_dict("records").__repr__()
        )

    # Analysis populations.
    specifications = [
        ("rate_primary_atleast1_hand", "rate", False),
        ("rate_sensitivity_both_hands", "rate", True),
        ("count_protocol_sensitivity_atleast1_hand", "count", False),
    ]

    all_attempt_rows = []
    population_rows = []
    desc_rows = []
    omnibus_rows = []
    pairwise_rows = []
    domain_corr_rows = []

    for label, mode, both in specifications:
        d = build_score(merged, mode=mode, require_both_hands=both)
        if not len(d):
            population_rows.append({
                "analysis": label,
                "attempts": 0,
                "participants": 0,
                "CON_attempts": 0,
                "ATG_attempts": 0,
                "MST_attempts": 0,
                "fit_status": "NO_ROWS",
            })
            continue

        result, fitdata, cov_type, stat, dft, p, pairs = fit_composite(
            d, "time_hand_eye_composite", label
        )

        population_rows.append({
            "analysis": label,
            "attempts": len(fitdata),
            "participants": fitdata["participant"].nunique(),
            "CON_attempts": int((fitdata["device"] == "CON").sum()),
            "ATG_attempts": int((fitdata["device"] == "ATG").sum()),
            "MST_attempts": int((fitdata["device"] == "MST").sum()),
            "fit_status": "PASS",
            "cov_type": cov_type,
        })

        for dev, g in fitdata.groupby("device"):
            desc_rows.append({
                "analysis": label,
                "device": dev,
                "attempts": len(g),
                "participants": g["participant"].nunique(),
                "mean_composite": g["time_hand_eye_composite"].mean(),
                "sd_composite": g["time_hand_eye_composite"].std(ddof=1),
                "median_composite": g["time_hand_eye_composite"].median(),
                "mean_time_component_z": g["time_component_z"].mean(),
                "mean_hand_activity_z": g["hand_activity_z"].mean(),
                "mean_eye_activity_z": g["eye_activity_z"].mean(),
            })

        omnibus_rows.append({
            "analysis": label,
            "wald_chi2_device": stat,
            "df": dft,
            "p": p,
            "cov_type": cov_type,
        })

        pairs.insert(0, "analysis", label)
        pairwise_rows.append(pairs)

        corr = fitdata[
            ["time_component_z", "hand_activity_z", "eye_activity_z"]
        ].corr()
        for a in corr.columns:
            for b in corr.columns:
                if a < b:
                    domain_corr_rows.append({
                        "analysis": label,
                        "domain_a": a,
                        "domain_b": b,
                        "pearson_r": corr.loc[a, b],
                    })

        keep = [
            "attempt_key", "participant", "device", "experience_stratum",
            "device_repetition_c", "chronological_attempt_sequence_c",
            "attempt_duration_sec",
            "fixation_count", "fixation_rate_per_second",
            "saccade_count", "saccade_rate_per_second",
            "velocity_peak_count_left", "velocity_peak_count_right",
            "hand_velocity_peak_rate_left", "hand_velocity_peak_rate_right",
            "time_component_z", "hand_activity_z", "eye_activity_z",
            "time_hand_eye_composite",
        ]
        tmp = fitdata[[c for c in keep if c in fitdata.columns]].copy()
        tmp.insert(0, "analysis", label)
        all_attempt_rows.append(tmp)

    attempts_out = (
        pd.concat(all_attempt_rows, ignore_index=True)
        if all_attempt_rows else pd.DataFrame()
    )
    populations = pd.DataFrame(population_rows)
    descriptives = pd.DataFrame(desc_rows)
    omnibus = pd.DataFrame(omnibus_rows)
    pairwise = (
        pd.concat(pairwise_rows, ignore_index=True)
        if pairwise_rows else pd.DataFrame()
    )
    domain_corr = pd.DataFrame(domain_corr_rows)

    attempts_out.to_csv(out / "time_hand_eye_composite_attempts.csv", index=False)
    populations.to_csv(out / "time_hand_eye_composite_population_summary.csv", index=False)
    descriptives.to_csv(out / "time_hand_eye_composite_device_descriptives.csv", index=False)
    omnibus.to_csv(out / "time_hand_eye_composite_device_omnibus.csv", index=False)
    pairwise.to_csv(out / "time_hand_eye_composite_pairwise.csv", index=False)
    domain_corr.to_csv(out / "time_hand_eye_composite_domain_correlations.csv", index=False)

    audit = {
        "frozen_master_attempts": int(meta["attempt_key"].nunique()),
        "fixation_attempts": int(fix["attempt_key"].nunique()),
        "saccade_attempts": int(sac["attempt_key"].nunique()),
        "eye_attempts_inner_overlap": int(eye["attempt_key"].nunique()),
        "eye_plus_frozen_master_overlap": int(
            meta.merge(eye[["attempt_key"]], on="attempt_key", how="inner")["attempt_key"].nunique()
        ),
        "attempts_with_at_least_one_main_hand": int(
            hand_wide.drop(columns=["attempt_key"]).notna().any(axis=1).sum()
        ),
        "definitions": {
            "primary_rate_composite": (
                "Equal-weight mean of z(log attempt duration), "
                "z-averaged available left/right hand velocity-peak rate, and "
                "z-averaged ultrasound-mapped fixation/saccade event rates."
            ),
            "both_hands_sensitivity": (
                "Same rate composite but requires both eligible GoPro >=60% hands."
            ),
            "count_protocol_sensitivity": (
                "Equal-weight count analogue using hand velocity-peak counts and "
                "eye fixation/saccade counts; secondary because counts are duration-dependent."
            ),
            "interpretation": (
                "Higher score means longer duration and/or greater observed hand/gaze-event activity; "
                "it is not a validated ease, workload, quality, or competence scale."
            ),
        },
    }
    (out / "time_hand_eye_composite_manifest.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    print("=== LPC TIME–HAND–EYE COMPOSITE ANALYSIS COMPLETE ===")
    print()
    print("Source overlap:")
    print(json.dumps({k: v for k, v in audit.items() if k != "definitions"}, indent=2))
    print()
    print("Analysis populations:")
    print(populations.to_string(index=False))
    print()
    print("Device omnibus tests:")
    print(omnibus.to_string(index=False))
    print()
    print("Pairwise contrasts:")
    print(pairwise.to_string(index=False) if len(pairwise) else "<none>")
    print()
    print("Primary interpretation: a higher score indicates longer procedural time")
    print("and/or greater hand / ultrasound-mapped eye-event activity. It is NOT")
    print("a validated ease-of-use, workload, competence, or quality score.")


if __name__ == "__main__":
    main()
