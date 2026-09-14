from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.stats.multitest import multipletests
except ImportError as e:
    raise SystemExit(
        "statsmodels is required. Install it in the project venv wit<SET_YOUR_ANALYSIS_ROOT>"
        "  python -m pip install statsmodels\n"
        f"Original import error: {e}"
    )

from scipy.stats import chi2, norm


OUTCOMES = {
    "mean_accel_norm_s2": {
        "transform": "log",
        "family": "gaussian",
        "effect_scale": "ratio_geometric_means",
    },
    "rms_accel_norm_s2": {
        "transform": "log",
        "family": "gaussian",
        "effect_scale": "ratio_geometric_means",
    },
    "peak_accel_norm_s2": {
        "transform": "log",
        "family": "gaussian",
        "effect_scale": "ratio_geometric_means",
    },
    "sparc": {
        "transform": "identity",
        "family": "gaussian",
        "effect_scale": "mean_difference",
    },
    "log_dimensionless_jerk": {
        "transform": "identity",
        "family": "gaussian",
        "effect_scale": "mean_difference",
    },
    "velocity_peak_count": {
        "transform": "identity",
        "family": "negative_binomial",
        "effect_scale": "rate_ratio",
        "offset": "observed_continuous_tracking_sec",
    },
    "path_length_norm": {
        "transform": "log",
        "family": "gaussian",
        "effect_scale": "ratio_geometric_means",
    },
    "path_efficiency": {
        "transform": "identity",
        "family": "gaussian",
        "effect_scale": "mean_difference",
    },
    "stillness_fraction": {
        "transform": "identity",
        "family": "gaussian",
        "effect_scale": "mean_difference",
    },
    "first_sustained_hand_movement_delay_sec": {
        "transform": "log1p",
        "family": "gaussian",
        "effect_scale": "ratio_1plus_delay",
    },
}


ANALYSIS_SETS = {
    "main_gopro_ge60": {
        "flag": "analysis_main",
        "confirmatory": True,
        "add_motion_source": False,
    },
    "highqc_gopro_ge80": {
        "flag": "analysis_highqc_sensitivity",
        "confirmatory": False,
        "add_motion_source": False,
    },
    "recovery_allsource_ge60": {
        "flag": "analysis_recovery_sensitivity",
        "confirmatory": False,
        "add_motion_source": True,
    },
}


PAIRWISE = [
    ("MST_vs_CON", "MST", "CON"),
    ("ATG_vs_CON", "ATG", "CON"),
    ("MST_vs_ATG", "MST", "ATG"),
]


def to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def transform_outcome(x: pd.Series, transform: str) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    if transform == "identity":
        return x
    if transform == "log":
        return np.log(x.where(x > 0))
    if transform == "log1p":
        return np.log1p(x.where(x >= 0))
    raise ValueError(transform)


def family_from_name(name: str):
    if name == "gaussian":
        return sm.families.Gaussian()
    if name == "negative_binomial":
        # Alpha=1 is a working variance choice; GEE robust/bias-reduced
        # covariance protects inference from variance misspecification.
        return sm.families.NegativeBinomial(alpha=1.0)
    if name == "poisson":
        return sm.families.Poisson()
    raise ValueError(name)


def device_param_name(index, level: str) -> str | None:
    hits = [
        str(name)
        for name in index
        if "C(device" in str(name) and f"[T.{level}]" in str(name)
    ]
    return hits[0] if len(hits) == 1 else None


def contrast_vector(result, high: str, low: str) -> np.ndarray:
    names = list(result.params.index)
    c = np.zeros(len(names), dtype=float)

    if high != "CON":
        name = device_param_name(names, high)
        if name is None:
            raise RuntimeError(f"Cannot find device coefficient for {high}.")
        c[names.index(name)] += 1.0

    if low != "CON":
        name = device_param_name(names, low)
        if name is None:
            raise RuntimeError(f"Cannot find device coefficient for {low}.")
        c[names.index(name)] -= 1.0

    return c


def contrast_stats(result, c: np.ndarray):
    beta = float(c @ result.params.to_numpy(float))
    cov = np.asarray(result.cov_params(), dtype=float)
    var = float(c @ cov @ c)
    se = math.sqrt(max(var, 0.0))
    z = beta / se if se > 0 else np.nan
    p = 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
    lo = beta - 1.959963984540054 * se if np.isfinite(se) else np.nan
    hi = beta + 1.959963984540054 * se if np.isfinite(se) else np.nan
    return beta, se, z, p, lo, hi


def omnibus_device_test(result):
    names = list(result.params.index)
    atm = device_param_name(names, "ATG")
    mst = device_param_name(names, "MST")
    if atm is None or mst is None:
        return np.nan, 2, np.nan

    idx = [names.index(atm), names.index(mst)]
    b = result.params.to_numpy(float)[idx]
    cov = np.asarray(result.cov_params(), dtype=float)[np.ix_(idx, idx)]

    try:
        stat = float(b.T @ np.linalg.pinv(cov) @ b)
        p = float(chi2.sf(stat, df=2))
    except Exception:
        stat = np.nan
        p = np.nan
    return stat, 2, p


def effect_scale(beta, lo, hi, effect_scale):
    if effect_scale in {"ratio_geometric_means", "rate_ratio", "ratio_1plus_delay"}:
        return math.exp(beta), math.exp(lo), math.exp(hi)
    return beta, lo, hi


def prepare_model_data(df, analysis_flag, hand, outcome, spec):
    sub = df[to_bool(df[analysis_flag]) & df["hand"].eq(hand)].copy()

    for c in ["device_repetition_c", "chronological_attempt_sequence_c"]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    sub["_y"] = transform_outcome(sub[outcome], spec["transform"])

    needed = [
        "_y",
        "participant",
        "device",
        "experience_stratum",
        "device_repetition_c",
        "chronological_attempt_sequence_c",
    ]

    offset = None
    if spec.get("offset"):
        expcol = spec["offset"]
        sub[expcol] = pd.to_numeric(sub[expcol], errors="coerce")
        sub["_offset"] = np.log(sub[expcol].where(sub[expcol] > 0))
        needed.append("_offset")

    sub = sub.replace([np.inf, -np.inf], np.nan)
    sub = sub.dropna(subset=needed).copy()

    return sub


def build_formula(add_motion_source: bool, hand: str) -> str:
    formula = (
        '_y ~ C(device, Treatment(reference="CON"))'
        ' + C(experience_stratum, Treatment(reference="None"))'
        ' + device_repetition_c'
        ' + chronological_attempt_sequence_c'
    )
    if add_motion_source:
        ref = "gopro_left" if hand == "left" else "gopro_right"
        formula += (
            f' + C(motion_source, Treatment(reference="{ref}"))'
        )
    return formula


def fit_gee(data, formula, family, offset=None):
    model = smf.gee(
        formula=formula,
        groups="participant",
        data=data,
        family=family,
        cov_struct=Exchangeable(),
        offset=offset,
    )

    # With ~15-22 participant clusters in some models, prefer the
    # Mancl-DeRouen bias-reduced covariance when supported.
    try:
        result = model.fit(cov_type="bias_reduced", maxiter=200)
        cov_type_used = "bias_reduced"
    except Exception:
        result = model.fit(cov_type="robust", maxiter=200)
        cov_type_used = "robust_fallback"

    return result, cov_type_used


def descriptives(df, flag, hand, outcome):
    sub = df[to_bool(df[flag]) & df["hand"].eq(hand)].copy()
    sub["_x"] = pd.to_numeric(sub[outcome], errors="coerce")

    rows = []
    for device, g in sub.groupby("device"):
        x = g["_x"].dropna()
        rows.append({
            "device": device,
            "rows": len(g),
            "nonmissing_n": len(x),
            "participants": g.loc[g["_x"].notna(), "participant"].nunique(),
            "mean": x.mean() if len(x) else np.nan,
            "sd": x.std(ddof=1) if len(x) > 1 else np.nan,
            "median": x.median() if len(x) else np.nan,
            "q25": x.quantile(.25) if len(x) else np.nan,
            "q75": x.quantile(.75) if len(x) else np.nan,
            "min": x.min() if len(x) else np.nan,
            "max": x.max() if len(x) else np.nan,
        })
    return rows


def fit_prespecified_models(df):
    status_rows = []
    omnibus_rows = []
    contrast_rows = []
    coef_rows = []
    descriptive_rows = []

    for set_name, set_spec in ANALYSIS_SETS.items():
        flag = set_spec["flag"]

        for hand in ["left", "right"]:
            for outcome, spec in OUTCOMES.items():

                for d in descriptives(df, flag, hand, outcome):
                    descriptive_rows.append({
                        "analysis_set": set_name,
                        "hand": hand,
                        "outcome": outcome,
                        **d,
                    })

                data = prepare_model_data(
                    df, flag, hand, outcome, spec
                )

                formula = build_formula(
                    set_spec["add_motion_source"], hand
                )

                status = {
                    "analysis_set": set_name,
                    "confirmatory": set_spec["confirmatory"],
                    "hand": hand,
                    "outcome": outcome,
                    "model_family": spec["family"],
                    "transform": spec["transform"],
                    "effect_scale": spec["effect_scale"],
                    "rows": len(data),
                    "participants": data["participant"].nunique(),
                    "device_levels": data["device"].nunique(),
                    "experience_levels": data["experience_stratum"].nunique(),
                    "formula": formula,
                    "fit_status": "",
                    "cov_type": "",
                    "converged": "",
                    "error": "",
                }

                if (
                    len(data) < 20
                    or data["participant"].nunique() < 8
                    or data["device"].nunique() < 3
                    or data["experience_stratum"].nunique() < 3
                ):
                    status["fit_status"] = "SKIPPED_NOT_ESTIMABLE"
                    status_rows.append(status)
                    continue

                family = family_from_name(spec["family"])
                offset = (
                    data["_offset"].to_numpy(float)
                    if spec.get("offset")
                    else None
                )

                try:
                    result, cov_type = fit_gee(
                        data, formula, family, offset=offset
                    )
                    status["fit_status"] = "PASS"
                    status["cov_type"] = cov_type
                    status["converged"] = bool(
                        getattr(result, "converged", True)
                    )

                    stat, df_test, p_omni = omnibus_device_test(result)
                    omnibus_rows.append({
                        "analysis_set": set_name,
                        "confirmatory": set_spec["confirmatory"],
                        "hand": hand,
                        "outcome": outcome,
                        "rows": len(data),
                        "participants": data["participant"].nunique(),
                        "device_wald_chi2": stat,
                        "device_df": df_test,
                        "device_omnibus_p": p_omni,
                        "device_omnibus_fdr_bh": np.nan,
                    })

                    for pname, high, low in PAIRWISE:
                        c = contrast_vector(result, high, low)
                        beta, se, z, p, lo, hi = contrast_stats(
                            result, c
                        )
                        eff, eff_lo, eff_hi = effect_scale(
                            beta, lo, hi, spec["effect_scale"]
                        )
                        contrast_rows.append({
                            "analysis_set": set_name,
                            "confirmatory": set_spec["confirmatory"],
                            "hand": hand,
                            "outcome": outcome,
                            "contrast": pname,
                            "high_device": high,
                            "low_device": low,
                            "model_coefficient": beta,
                            "model_se": se,
                            "z": z,
                            "p_raw": p,
                            "p_holm_within_model": np.nan,
                            "ci95_low_model_scale": lo,
                            "ci95_high_model_scale": hi,
                            "effect_scale": spec["effect_scale"],
                            "effect_estimate": eff,
                            "effect_ci95_low": eff_lo,
                            "effect_ci95_high": eff_hi,
                        })

                    for term in result.params.index:
                        beta = float(result.params[term])
                        se = float(result.bse[term])
                        z = beta / se if se > 0 else np.nan
                        p = 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
                        coef_rows.append({
                            "analysis_set": set_name,
                            "hand": hand,
                            "outcome": outcome,
                            "term": str(term),
                            "coefficient": beta,
                            "se": se,
                            "z": z,
                            "p": p,
                        })

                except Exception as e:
                    status["fit_status"] = "FAIL"
                    status["error"] = repr(e)

                status_rows.append(status)

    status = pd.DataFrame(status_rows)
    omnibus = pd.DataFrame(omnibus_rows)
    contrasts = pd.DataFrame(contrast_rows)
    coefficients = pd.DataFrame(coef_rows)
    descriptives_df = pd.DataFrame(descriptive_rows)

    # BH FDR across confirmatory MAIN outcome x hand omnibus device tests only.
    if len(omnibus):
        mask = (
            omnibus["confirmatory"].astype(bool)
            & omnibus["device_omnibus_p"].notna()
        )
        if mask.any():
            _, q, _, _ = multipletests(
                omnibus.loc[mask, "device_omnibus_p"].astype(float),
                alpha=0.05,
                method="fdr_bh",
            )
            omnibus.loc[mask, "device_omnibus_fdr_bh"] = q

    # Holm across the three pairwise device contrasts within each
    # analysis-set x hand x outcome model.
    if len(contrasts):
        for _, idx in contrasts.groupby(
            ["analysis_set", "hand", "outcome"]
        ).groups.items():
            idx = list(idx)
            p = pd.to_numeric(
                contrasts.loc[idx, "p_raw"], errors="coerce"
            )
            valid = p.notna()
            if valid.any():
                _, adj, _, _ = multipletests(
                    p[valid].astype(float),
                    alpha=0.05,
                    method="holm",
                )
                contrasts.loc[
                    [idx[i] for i, ok in enumerate(valid) if ok],
                    "p_holm_within_model",
                ] = adj

    # Merge main omnibus FDR onto main pairwise rows to make the
    # confirmatory gate explicit.
    if len(contrasts) and len(omnibus):
        gate = omnibus[
            [
                "analysis_set", "hand", "outcome",
                "device_omnibus_p", "device_omnibus_fdr_bh"
            ]
        ]
        contrasts = contrasts.merge(
            gate,
            on=["analysis_set", "hand", "outcome"],
            how="left",
            validate="many_to_one",
        )
        contrasts["confirmatory_pairwise_gate_pass"] = (
            (contrasts["analysis_set"] == "main_gopro_ge60")
            & pd.to_numeric(
                contrasts["device_omnibus_fdr_bh"],
                errors="coerce"
            ).le(0.05)
        )

    return status, omnibus, contrasts, coefficients, descriptives_df


def fit_main_model_variants(df):
    """Pre-specified family/transform sensitivity models for MAIN only."""
    rows = []

    variant_specs = []

    for outcome in [
        "mean_accel_norm_s2",
        "rms_accel_norm_s2",
        "peak_accel_norm_s2",
        "path_length_norm",
        "first_sustained_hand_movement_delay_sec",
    ]:
        variant_specs.append(
            (outcome, "raw_gaussian_sensitivity", "identity", "gaussian", None)
        )

    variant_specs.append(
        ("velocity_peak_count", "poisson_rate_sensitivity",
         "identity", "poisson", "observed_continuous_tracking_sec")
    )

    for outcome in ["path_efficiency", "stillness_fraction"]:
        variant_specs.append(
            (outcome, "logit_gaussian_nonboundary_sensitivity",
             "logit", "gaussian", None)
        )

    for hand in ["left", "right"]:
        for outcome, variant, transform, family_name, offset_col in variant_specs:
            sub = df[
                to_bool(df["analysis_main"])
                & df["hand"].eq(hand)
            ].copy()

            for c in ["device_repetition_c", "chronological_attempt_sequence_c"]:
                sub[c] = pd.to_numeric(sub[c], errors="coerce")

            x = pd.to_numeric(sub[outcome], errors="coerce")
            if transform == "identity":
                sub["_y"] = x
            elif transform == "logit":
                valid = (x > 0) & (x < 1)
                sub["_y"] = np.nan
                sub.loc[valid, "_y"] = np.log(
                    x[valid] / (1 - x[valid])
                )
            else:
                raise ValueError(transform)

            needed = [
                "_y", "participant", "device", "experience_stratum",
                "device_repetition_c", "chronological_attempt_sequence_c"
            ]
            if offset_col:
                sub[offset_col] = pd.to_numeric(
                    sub[offset_col], errors="coerce"
                )
                sub["_offset"] = np.log(
                    sub[offset_col].where(sub[offset_col] > 0)
                )
                needed.append("_offset")

            sub = sub.replace([np.inf, -np.inf], np.nan).dropna(
                subset=needed
            )

            outrow = {
                "hand": hand,
                "outcome": outcome,
                "variant": variant,
                "rows": len(sub),
                "participants": sub["participant"].nunique(),
                "fit_status": "",
                "device_omnibus_p": np.nan,
                "error": "",
            }

            if (
                len(sub) < 20
                or sub["participant"].nunique() < 8
                or sub["device"].nunique() < 3
                or sub["experience_stratum"].nunique() < 3
            ):
                outrow["fit_status"] = "SKIPPED_NOT_ESTIMABLE"
                rows.append(outrow)
                continue

            formula = build_formula(False, hand)

            try:
                family = family_from_name(family_name)
                offset = (
                    sub["_offset"].to_numpy(float)
                    if offset_col else None
                )
                result, cov_type = fit_gee(
                    sub, formula, family, offset=offset
                )
                _, _, p = omnibus_device_test(result)
                outrow["fit_status"] = "PASS"
                outrow["cov_type"] = cov_type
                outrow["device_omnibus_p"] = p
            except Exception as e:
                outrow["fit_status"] = "FAIL"
                outrow["error"] = repr(e)

            rows.append(outrow)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run the frozen SIMETRIC scalar hand-motion repeated-measures GEE analysis. "
            "Fits left/right separately, clusters by participant, applies the frozen "
            "covariates/transforms, controls main omnibus multiplicity by BH FDR, and "
            "Holm-adjusts the three pairwise device contrasts within each model."
        )
    )
    ap.add_argument("--master", required=True)
    ap.add_argument("--model-spec", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Preserve literal "None" experience category.
    df = pd.read_csv(
        args.master,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )

    spec_path = Path(args.model_spec)
    if not spec_path.exists():
        raise FileNotFoundError(spec_path)
    frozen_spec = json.loads(spec_path.read_text(encoding="utf-8"))

    valid_experience = {"Higher", "Some", "None"}
    if set(df["experience_stratum"].unique()) - valid_experience:
        raise RuntimeError(
            "Unexpected experience labels in analysis master: "
            + repr(sorted(set(df["experience_stratum"].unique()) - valid_experience))
        )

    status, omnibus, contrasts, coefficients, descriptives_df = (
        fit_prespecified_models(df)
    )
    variants = fit_main_model_variants(df)

    status_path = outdir / "scalar_gee_model_status.csv"
    omnibus_path = outdir / "scalar_gee_device_omnibus_tests.csv"
    contrast_path = outdir / "scalar_gee_device_pairwise_contrasts.csv"
    coef_path = outdir / "scalar_gee_coefficients.csv"
    desc_path = outdir / "scalar_gee_descriptives_by_device.csv"
    variant_path = outdir / "scalar_gee_main_model_variant_sensitivity.csv"

    status.to_csv(status_path, index=False)
    omnibus.to_csv(omnibus_path, index=False)
    contrasts.to_csv(contrast_path, index=False)
    coefficients.to_csv(coef_path, index=False)
    descriptives_df.to_csv(desc_path, index=False)
    variants.to_csv(variant_path, index=False)

    analysis_manifest = {
        "input_master": str(Path(args.master)),
        "input_model_spec": str(spec_path),
        "frozen_model_specification": frozen_spec,
        "covariance": (
            "bias_reduced GEE covariance attempted first because hand-specific "
            "models contain roughly 15-22 participant clusters; robust covariance "
            "used only as fallback if bias-reduced fitting fails"
        ),
        "negative_binomial_working_alpha": 1.0,
        "main_multiplicity": (
            "Benjamini-Hochberg FDR across all main outcome x hand 2-df "
            "omnibus device tests"
        ),
        "pairwise_multiplicity": (
            "Holm adjustment across MST-vs-CON, ATG-vs-CON, MST-vs-ATG "
            "within each analysis-set x outcome x hand model"
        ),
        "interpretation_gate": (
            "Main pairwise contrasts are confirmatory only when the corresponding "
            "main outcome x hand omnibus device test survives BH FDR <=0.05."
        ),
        "sensitivity_rule": (
            "High-QC and recovery analyses assess robustness and do not rescue "
            "a non-confirmatory main result."
        ),
    }
    manifest_path = outdir / "scalar_gee_analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(analysis_manifest, indent=2),
        encoding="utf-8",
    )

    print("=== SCALAR GEE ANALYSIS COMPLETE ===")
    print()
    print("Model status:")
    print(
        status.groupby(
            ["analysis_set", "fit_status"], dropna=False
        ).size().reset_index(name="models").to_string(index=False)
    )

    print()
    print("=== MAIN DEVICE OMNIBUS TESTS ===")
    main_omni = omnibus[
        omnibus["analysis_set"].eq("main_gopro_ge60")
    ].copy()
    if len(main_omni):
        print(
            main_omni[
                [
                    "hand", "outcome", "rows", "participants",
                    "device_wald_chi2", "device_omnibus_p",
                    "device_omnibus_fdr_bh",
                ]
            ].sort_values(
                ["device_omnibus_fdr_bh", "device_omnibus_p"],
                na_position="last"
            ).to_string(index=False)
        )
    else:
        print("No successful main omnibus models.")

    print()
    print("=== CONFIRMATORY MAIN PAIRWISE CONTRASTS ===")
    gated = contrasts[
        contrasts["confirmatory_pairwise_gate_pass"].fillna(False)
    ].copy()
    if len(gated):
        print(
            gated[
                [
                    "hand", "outcome", "contrast",
                    "effect_scale", "effect_estimate",
                    "effect_ci95_low", "effect_ci95_high",
                    "p_raw", "p_holm_within_model",
                    "device_omnibus_fdr_bh",
                ]
            ].to_string(index=False)
        )
    else:
        print(
            "None. This means no main outcome x hand omnibus device test "
            "passed the pre-specified BH-FDR gate, or no gated pairwise models "
            "were available."
        )

    print()
    print("Main model-variant sensitivity status:")
    print(
        variants.groupby(["variant", "fit_status"], dropna=False)
        .size().reset_index(name="models")
        .to_string(index=False)
    )

    print()
    print("Wrote:", status_path)
    print("Wrote:", omnibus_path)
    print("Wrote:", contrast_path)
    print("Wrote:", coef_path)
    print("Wrote:", desc_path)
    print("Wrote:", variant_path)
    print("Wrote:", manifest_path)
    print()
    print(
        "Do not interpret device effects from the console alone. "
        "Use the omnibus table as the confirmatory gate, then the pairwise "
        "effect estimates/95% CIs with Holm-adjusted p-values. Sensitivity "
        "analyses assess robustness only."
    )


if __name__ == "__main__":
    main()
