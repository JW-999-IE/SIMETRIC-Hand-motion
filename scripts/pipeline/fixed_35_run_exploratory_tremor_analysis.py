from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2, norm


PAIRWISE = [
    ("MST_vs_CON", "MST", "CON"),
    ("ATG_vs_CON", "ATG", "CON"),
    ("MST_vs_ATG", "MST", "ATG"),
]


def as_bool(s):
    return s.astype(str).str.lower().isin({"true", "1", "yes"})


def device_name(names, level):
    hits = [n for n in names if "C(device" in str(n) and f"[T.{level}]" in str(n)]
    return hits[0] if len(hits) == 1 else None


def contrast(result, high, low):
    names = list(result.params.index)
    c = np.zeros(len(names))
    if high != "CON":
        c[names.index(device_name(names, high))] += 1
    if low != "CON":
        c[names.index(device_name(names, low))] -= 1
    b = float(c @ result.params.to_numpy(float))
    cov = np.asarray(result.cov_params(), float)
    se = math.sqrt(max(float(c @ cov @ c), 0))
    p = 2 * norm.sf(abs(b / se)) if se > 0 else np.nan
    return {
        "ratio": math.exp(b),
        "ci95_low": math.exp(b - 1.95996398454 * se),
        "ci95_high": math.exp(b + 1.95996398454 * se),
        "p_raw": p,
    }


def omnibus(result):
    names = list(result.params.index)
    idx = [names.index(device_name(names, x)) for x in ["ATG", "MST"]]
    b = result.params.to_numpy(float)[idx]
    cov = np.asarray(result.cov_params(), float)[np.ix_(idx, idx)]
    stat = float(b.T @ np.linalg.pinv(cov) @ b)
    return stat, float(chi2.sf(stat, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.master, dtype=str, keep_default_na=False, na_filter=False)

    sets = {
        "main_gopro_ge60": "analysis_main",
        "highqc_gopro_ge80": "analysis_highqc_sensitivity",
    }

    omni_rows, pair_rows, desc_rows, status_rows = [], [], [], []

    for set_name, flag in sets.items():
        for hand in ["left", "right"]:
            d = df[as_bool(df[flag]) & df["hand"].eq(hand)].copy()
            d["tremor_rms_norm"] = pd.to_numeric(d["tremor_rms_norm"], errors="coerce")
            d["tremor_dominant_hz"] = pd.to_numeric(d["tremor_dominant_hz"], errors="coerce")
            d["device_repetition_c"] = pd.to_numeric(d["device_repetition_c"], errors="coerce")
            d["chronological_attempt_sequence_c"] = pd.to_numeric(
                d["chronological_attempt_sequence_c"], errors="coerce"
            )

            for dev, g in d.groupby("device"):
                x = g["tremor_rms_norm"].dropna()
                hz = g["tremor_dominant_hz"].dropna()
                desc_rows.append({
                    "analysis_set": set_name,
                    "hand": hand,
                    "device": dev,
                    "n_tremor_rms": len(x),
                    "participants": g.loc[g["tremor_rms_norm"].notna(), "participant"].nunique(),
                    "median_tremor_rms_norm": x.median() if len(x) else np.nan,
                    "q25_tremor_rms_norm": x.quantile(.25) if len(x) else np.nan,
                    "q75_tremor_rms_norm": x.quantile(.75) if len(x) else np.nan,
                    "median_dominant_hz": hz.median() if len(hz) else np.nan,
                })

            d = d[d["tremor_rms_norm"] > 0].dropna(
                subset=[
                    "tremor_rms_norm", "participant", "device", "experience_stratum",
                    "device_repetition_c", "chronological_attempt_sequence_c",
                ]
            )
            if len(d) < 20 or d["participant"].nunique() < 8 or d["device"].nunique() < 3:
                status_rows.append({
                    "analysis_set": set_name, "hand": hand,
                    "fit_status": "SKIPPED_NOT_ESTIMABLE",
                    "rows": len(d), "participants": d["participant"].nunique(),
                })
                continue

            d["_y"] = np.log(d["tremor_rms_norm"])
            formula = (
                '_y ~ C(device, Treatment(reference="CON"))'
                ' + C(experience_stratum, Treatment(reference="None"))'
                ' + device_repetition_c + chronological_attempt_sequence_c'
            )
            model = smf.gee(
                formula=formula, groups="participant", data=d,
                family=sm.families.Gaussian(), cov_struct=Exchangeable()
            )
            try:
                r = model.fit(cov_type="bias_reduced")
                cov_type = "bias_reduced"
            except Exception:
                r = model.fit(cov_type="robust")
                cov_type = "robust_fallback"

            stat, p = omnibus(r)
            omni_rows.append({
                "analysis_set": set_name,
                "hand": hand,
                "rows": len(d),
                "participants": d["participant"].nunique(),
                "wald_chi2": stat,
                "p_raw": p,
                "p_holm_main_two_hands": np.nan,
            })
            for label, high, low in PAIRWISE:
                z = contrast(r, high, low)
                pair_rows.append({
                    "analysis_set": set_name, "hand": hand,
                    "contrast": label, **z,
                    "p_holm_within_hand": np.nan,
                })
            status_rows.append({
                "analysis_set": set_name, "hand": hand,
                "fit_status": "PASS", "rows": len(d),
                "participants": d["participant"].nunique(),
                "cov_type": cov_type,
            })

    omni = pd.DataFrame(omni_rows)
    pair = pd.DataFrame(pair_rows)
    desc = pd.DataFrame(desc_rows)
    status = pd.DataFrame(status_rows)

    if len(omni):
        mask = omni["analysis_set"].eq("main_gopro_ge60")
        if mask.any():
            _, adj, _, _ = multipletests(omni.loc[mask, "p_raw"], method="holm")
            omni.loc[mask, "p_holm_main_two_hands"] = adj

    if len(pair):
        for _, idx in pair.groupby(["analysis_set", "hand"]).groups.items():
            idx = list(idx)
            _, adj, _, _ = multipletests(pair.loc[idx, "p_raw"], method="holm")
            pair.loc[idx, "p_holm_within_hand"] = adj

    status.to_csv(out / "tremor_model_status.csv", index=False)
    desc.to_csv(out / "tremor_descriptives.csv", index=False)
    omni.to_csv(out / "tremor_device_omnibus.csv", index=False)
    pair.to_csv(out / "tremor_device_pairwise.csv", index=False)

    print("=== EXPLORATORY LPC TREMOR-LIKE MOTION ANALYSIS COMPLETE ===")
    print("IMPORTANT: this is tracking-jitter-sensitive high-frequency residual motion, not a physiological tremor diagnosis.")
    print()
    print("Main omnibus:")
    print(omni[omni["analysis_set"].eq("main_gopro_ge60")].to_string(index=False))
    print()
    print("Main pairwise:")
    print(pair[pair["analysis_set"].eq("main_gopro_ge60")].to_string(index=False))

if __name__ == "__main__":
    main()
