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
from scipy.stats import chi2


OUTCOMES = {
    "mean_accel_norm_s2": ("log", "gaussian"),
    "rms_accel_norm_s2": ("log", "gaussian"),
    "peak_accel_norm_s2": ("log", "gaussian"),
    "sparc": ("identity", "gaussian"),
    "log_dimensionless_jerk": ("identity", "gaussian"),
    "velocity_peak_count": ("identity", "negative_binomial"),
    "path_length_norm": ("log", "gaussian"),
    "path_efficiency": ("identity", "gaussian"),
    "stillness_fraction": ("identity", "gaussian"),
    "first_sustained_hand_movement_delay_sec": ("log1p", "gaussian"),
}


def as_bool(s):
    return s.astype(str).str.lower().isin({"true", "1", "yes"})


def transform(x, t):
    x = pd.to_numeric(x, errors="coerce")
    if t == "log": return np.log(x.where(x > 0))
    if t == "log1p": return np.log1p(x.where(x >= 0))
    return x


def experience_indices(result):
    names = list(result.params.index)
    return [
        i for i, n in enumerate(names)
        if "C(experience_stratum" in str(n)
        and ("[T.Higher]" in str(n) or "[T.Some]" in str(n))
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.master, dtype=str, keep_default_na=False, na_filter=False)
    df = df[as_bool(df["analysis_main"])].copy()

    rows = []
    coef_rows = []
    for hand in ["left", "right"]:
        for outcome, (trans, fam) in OUTCOMES.items():
            d = df[df["hand"].eq(hand)].copy()
            d["_y"] = transform(d[outcome], trans)
            d["device_repetition_c"] = pd.to_numeric(d["device_repetition_c"], errors="coerce")
            d["chronological_attempt_sequence_c"] = pd.to_numeric(d["chronological_attempt_sequence_c"], errors="coerce")

            offset = None
            if outcome == "velocity_peak_count":
                exp = pd.to_numeric(d["observed_continuous_tracking_sec"], errors="coerce")
                d["_offset"] = np.log(exp.where(exp > 0))
                offset = d["_offset"]

            d = d.replace([np.inf, -np.inf], np.nan).dropna(
                subset=["_y", "participant", "device", "experience_stratum",
                        "device_repetition_c", "chronological_attempt_sequence_c"]
                + (["_offset"] if outcome == "velocity_peak_count" else [])
            )
            if len(d) < 20 or d["experience_stratum"].nunique() < 3:
                continue

            family = (
                sm.families.NegativeBinomial(alpha=1.0)
                if fam == "negative_binomial" else sm.families.Gaussian()
            )
            formula = (
                '_y ~ C(device, Treatment(reference="CON"))'
                ' + C(experience_stratum, Treatment(reference="None"))'
                ' + device_repetition_c + chronological_attempt_sequence_c'
            )
            model = smf.gee(
                formula=formula, groups="participant", data=d,
                family=family, cov_struct=Exchangeable(),
                offset=(d["_offset"].to_numpy(float) if outcome == "velocity_peak_count" else None),
            )
            try:
                r = model.fit(cov_type="bias_reduced")
                cov_type = "bias_reduced"
            except Exception:
                r = model.fit(cov_type="robust")
                cov_type = "robust_fallback"

            ix = experience_indices(r)
            b = r.params.to_numpy(float)[ix]
            cov = np.asarray(r.cov_params(), float)[np.ix_(ix, ix)]
            stat = float(b.T @ np.linalg.pinv(cov) @ b)
            p = float(chi2.sf(stat, len(ix)))
            rows.append({
                "hand": hand,
                "outcome": outcome,
                "rows": len(d),
                "participants": d["participant"].nunique(),
                "wald_chi2_experience": stat,
                "df": len(ix),
                "p_raw": p,
                "p_fdr_bh_across_20_tests": np.nan,
                "cov_type": cov_type,
            })

            for term in r.params.index:
                if "C(experience_stratum" in str(term):
                    bb = float(r.params[term])
                    se = float(r.bse[term])
                    if trans in {"log", "log1p"} or fam == "negative_binomial":
                        effect = math.exp(bb)
                        lo = math.exp(bb - 1.95996398454 * se)
                        hi = math.exp(bb + 1.95996398454 * se)
                        scale = "ratio"
                    else:
                        effect = bb
                        lo = bb - 1.95996398454 * se
                        hi = bb + 1.95996398454 * se
                        scale = "mean_difference"
                    coef_rows.append({
                        "hand": hand, "outcome": outcome, "term": str(term),
                        "effect_scale": scale, "effect_estimate": effect,
                        "ci95_low": lo, "ci95_high": hi,
                        "p_raw": float(r.pvalues[term]),
                    })

    omni = pd.DataFrame(rows)
    coefs = pd.DataFrame(coef_rows)
    if len(omni):
        _, adj, _, _ = multipletests(omni["p_raw"], method="fdr_bh")
        omni["p_fdr_bh_across_20_tests"] = adj

    omni.to_csv(out / "prior_lpc_experience_hand_omnibus.csv", index=False)
    coefs.to_csv(out / "prior_lpc_experience_hand_coefficients.csv", index=False)

    print("=== PRIOR LPC EXPERIENCE × HAND-MOTION OUTCOME ANALYSIS COMPLETE ===")
    print("These are participant-level experience associations, not within-person device effects.")
    print()
    print(omni.sort_values("p_fdr_bh_across_20_tests").to_string(index=False))
