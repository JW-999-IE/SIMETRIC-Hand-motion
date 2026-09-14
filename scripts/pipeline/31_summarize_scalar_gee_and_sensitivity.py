from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


NULL_BY_SCALE = {
    "ratio_geometric_means": 1.0,
    "rate_ratio": 1.0,
    "ratio_1plus_delay": 1.0,
    "mean_difference": 0.0,
}


def direction(effect: float, null: float) -> str:
    if not np.isfinite(effect):
        return "missing"
    if effect > null:
        return "positive"
    if effect < null:
        return "negative"
    return "null"


def ci_excludes_null(lo: float, hi: float, null: float) -> bool:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return False
    return (lo > null) or (hi < null)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Summarize confirmatory scalar GEE findings and compare their direction/"
            "precision across the prespecified high-QC and recovery sensitivity sets."
        )
    )
    ap.add_argument("--analysis-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    analysis_dir = Path(args.analysis_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    omni = pd.read_csv(
        analysis_dir / "scalar_gee_device_omnibus_tests.csv",
        keep_default_na=False,
    )
    pair = pd.read_csv(
        analysis_dir / "scalar_gee_device_pairwise_contrasts.csv",
        keep_default_na=False,
    )
    variants = pd.read_csv(
        analysis_dir / "scalar_gee_main_model_variant_sensitivity.csv",
        keep_default_na=False,
    )
    status = pd.read_csv(
        analysis_dir / "scalar_gee_model_status.csv",
        keep_default_na=False,
    )

    # Numeric conversion.
    for c in [
        "device_omnibus_p",
        "device_omnibus_fdr_bh",
        "device_wald_chi2",
    ]:
        if c in omni.columns:
            omni[c] = pd.to_numeric(omni[c], errors="coerce")

    for c in [
        "effect_estimate",
        "effect_ci95_low",
        "effect_ci95_high",
        "p_raw",
        "p_holm_within_model",
        "device_omnibus_p",
        "device_omnibus_fdr_bh",
    ]:
        if c in pair.columns:
            pair[c] = pd.to_numeric(pair[c], errors="coerce")

    if "device_omnibus_p" in variants.columns:
        variants["device_omnibus_p"] = pd.to_numeric(
            variants["device_omnibus_p"], errors="coerce"
        )

    main_omni = omni[
        omni["analysis_set"].eq("main_gopro_ge60")
    ].copy()
    main_omni["main_fdr_pass"] = (
        main_omni["device_omnibus_fdr_bh"].le(0.05)
    )

    confirmatory_omni = main_omni[
        main_omni["main_fdr_pass"]
    ].copy()

    null_omni = main_omni[
        ~main_omni["main_fdr_pass"]
    ].copy()

    confirmatory_omni_path = outdir / "main_confirmatory_omnibus_findings.csv"
    null_omni_path = outdir / "main_nonconfirmatory_omnibus_findings.csv"
    confirmatory_omni.to_csv(confirmatory_omni_path, index=False)
    null_omni.to_csv(null_omni_path, index=False)

    # Main pairwise findings, gated by omnibus FDR and pairwise Holm.
    main_pair = pair[
        pair["analysis_set"].eq("main_gopro_ge60")
    ].copy()

    main_pair["omnibus_gate_pass"] = (
        main_pair["device_omnibus_fdr_bh"].le(0.05)
    )
    main_pair["holm_pairwise_pass"] = (
        main_pair["p_holm_within_model"].le(0.05)
    )
    main_pair["confirmatory_pairwise"] = (
        main_pair["omnibus_gate_pass"]
        & main_pair["holm_pairwise_pass"]
    )

    confirmatory_pair = main_pair[
        main_pair["confirmatory_pairwise"]
    ].copy()

    confirmatory_pair_path = outdir / "main_confirmatory_pairwise_findings.csv"
    confirmatory_pair.to_csv(confirmatory_pair_path, index=False)

    # Sensitivity concordance for confirmatory main pairwise findings.
    sensitivity_sets = [
        "highqc_gopro_ge80",
        "recovery_allsource_ge60",
    ]

    sensitivity_rows = []

    for _, m in confirmatory_pair.iterrows():
        scale = str(m["effect_scale"])
        null = NULL_BY_SCALE.get(scale, 0.0)

        main_eff = float(m["effect_estimate"])
        main_lo = float(m["effect_ci95_low"])
        main_hi = float(m["effect_ci95_high"])
        main_dir = direction(main_eff, null)

        row = {
            "hand": m["hand"],
            "outcome": m["outcome"],
            "contrast": m["contrast"],
            "effect_scale": scale,
            "null_value": null,
            "main_effect": main_eff,
            "main_ci_low": main_lo,
            "main_ci_high": main_hi,
            "main_holm_p": float(m["p_holm_within_model"]),
            "main_direction": main_dir,
        }

        all_direction_concordant = True
        all_ci_exclude_null = True

        for aset in sensitivity_sets:
            s = pair[
                pair["analysis_set"].eq(aset)
                & pair["hand"].eq(m["hand"])
                & pair["outcome"].eq(m["outcome"])
                & pair["contrast"].eq(m["contrast"])
            ]

            prefix = (
                "highqc"
                if aset == "highqc_gopro_ge80"
                else "recovery"
            )

            if len(s) != 1:
                row[f"{prefix}_available"] = False
                row[f"{prefix}_effect"] = np.nan
                row[f"{prefix}_ci_low"] = np.nan
                row[f"{prefix}_ci_high"] = np.nan
                row[f"{prefix}_holm_p"] = np.nan
                row[f"{prefix}_direction_concordant"] = False
                row[f"{prefix}_ci_excludes_null"] = False
                all_direction_concordant = False
                all_ci_exclude_null = False
                continue

            s = s.iloc[0]
            eff = float(s["effect_estimate"])
            lo = float(s["effect_ci95_low"])
            hi = float(s["effect_ci95_high"])
            sdir = direction(eff, null)
            dir_ok = sdir == main_dir
            ci_ok = ci_excludes_null(lo, hi, null)

            row[f"{prefix}_available"] = True
            row[f"{prefix}_effect"] = eff
            row[f"{prefix}_ci_low"] = lo
            row[f"{prefix}_ci_high"] = hi
            row[f"{prefix}_holm_p"] = float(s["p_holm_within_model"])
            row[f"{prefix}_direction"] = sdir
            row[f"{prefix}_direction_concordant"] = dir_ok
            row[f"{prefix}_ci_excludes_null"] = ci_ok

            all_direction_concordant &= dir_ok
            all_ci_exclude_null &= ci_ok

        row["both_sensitivity_directions_concordant"] = (
            all_direction_concordant
        )
        row["both_sensitivity_cis_exclude_null"] = (
            all_ci_exclude_null
        )

        if all_direction_concordant and all_ci_exclude_null:
            robustness = "STRONG"
        elif all_direction_concordant:
            robustness = "DIRECTIONALLY_CONCORDANT"
        else:
            robustness = "INCONSISTENT_DIRECTION"

        row["robustness_classification"] = robustness
        sensitivity_rows.append(row)

    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity_path = outdir / "confirmatory_pairwise_sensitivity_concordance.csv"
    sensitivity.to_csv(sensitivity_path, index=False)

    # Main model-form sensitivity: descriptive only, not a new significance gate.
    variant_path = outdir / "main_model_form_sensitivity_summary.csv"
    variants.to_csv(variant_path, index=False)

    # Compact effect summary with human-readable percentage-point / percent effects.
    readable_rows = []
    for _, r in confirmatory_pair.iterrows():
        scale = str(r["effect_scale"])
        eff = float(r["effect_estimate"])
        lo = float(r["effect_ci95_low"])
        hi = float(r["effect_ci95_high"])

        if scale in {
            "ratio_geometric_means",
            "rate_ratio",
            "ratio_1plus_delay",
        }:
            readable_effect = 100.0 * (eff - 1.0)
            readable_lo = 100.0 * (lo - 1.0)
            readable_hi = 100.0 * (hi - 1.0)
            readable_unit = "percent_change"
        else:
            readable_effect = 100.0 * eff if "fraction" in str(r["outcome"]) else eff
            readable_lo = 100.0 * lo if "fraction" in str(r["outcome"]) else lo
            readable_hi = 100.0 * hi if "fraction" in str(r["outcome"]) else hi
            readable_unit = (
                "percentage_points"
                if "fraction" in str(r["outcome"])
                else "absolute_difference"
            )

        readable_rows.append({
            "hand": r["hand"],
            "outcome": r["outcome"],
            "contrast": r["contrast"],
            "effect_scale": scale,
            "effect_estimate": eff,
            "ci95_low": lo,
            "ci95_high": hi,
            "holm_p": float(r["p_holm_within_model"]),
            "omnibus_fdr_q": float(r["device_omnibus_fdr_bh"]),
            "readable_effect": readable_effect,
            "readable_ci_low": readable_lo,
            "readable_ci_high": readable_hi,
            "readable_unit": readable_unit,
        })

    readable = pd.DataFrame(readable_rows)
    readable_path = outdir / "main_confirmatory_effects_readable.csv"
    readable.to_csv(readable_path, index=False)

    # Markdown report.
    report = []
    report.append("# SIMETRIC scalar hand-motion GEE summary")
    report.append("")
    report.append(
        f"Main confirmatory omnibus device findings: "
        f"{len(confirmatory_omni)} of {len(main_omni)} hand×outcome tests "
        f"passed BH-FDR q<=0.05."
    )
    report.append("")
    report.append("## Confirmatory omnibus findings")
    report.append("")
    for _, r in confirmatory_omni.sort_values(
        ["hand", "device_omnibus_fdr_bh", "outcome"]
    ).iterrows():
        report.append(
            f"- {r['hand']} — {r['outcome']}: "
            f"Wald χ²={r['device_wald_chi2']:.3f}, "
            f"p={r['device_omnibus_p']:.4g}, "
            f"BH-FDR q={r['device_omnibus_fdr_bh']:.4g}."
        )

    report.append("")
    report.append("## Confirmatory pairwise findings")
    report.append("")
    for _, r in readable.sort_values(
        ["hand", "outcome", "contrast"]
    ).iterrows():
        report.append(
            f"- {r['hand']} — {r['outcome']} — {r['contrast']}: "
            f"effect={r['effect_estimate']:.4g} "
            f"(95% CI {r['ci95_low']:.4g} to {r['ci95_high']:.4g}), "
            f"Holm p={r['holm_p']:.4g}."
        )

    report.append("")
    report.append("## Interpretation rules")
    report.append("")
    report.append(
        "- Only main GoPro>=60% omnibus tests passing BH-FDR are confirmatory."
    )
    report.append(
        "- Pairwise contrasts are confirmatory only when the omnibus gate passes "
        "and their within-model Holm-adjusted p<=0.05."
    )
    report.append(
        "- High-QC and all-source analyses are robustness checks and cannot "
        "convert a non-confirmatory main result into a confirmatory one."
    )
    report.append(
        "- Grip remains screening-only; tremor remains exploratory; trajectory "
        "requires a separate longitudinal/functional analysis."
    )

    report_path = outdir / "scalar_gee_summary.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("=== SCALAR GEE SYNTHESIS COMPLETE ===")
    print(
        f"Main omnibus tests passing BH-FDR: "
        f"{len(confirmatory_omni)} / {len(main_omni)}"
    )
    print(
        f"Confirmatory pairwise contrasts after omnibus gate + Holm: "
        f"{len(confirmatory_pair)}"
    )

    if len(sensitivity):
        print()
        print("Sensitivity robustness classifications:")
        print(
            sensitivity["robustness_classification"]
            .value_counts(dropna=False)
            .to_string()
        )

    print()
    print("Confirmatory omnibus findings:")
    print(
        confirmatory_omni[
            [
                "hand", "outcome",
                "device_omnibus_p", "device_omnibus_fdr_bh"
            ]
        ].sort_values(
            ["hand", "device_omnibus_fdr_bh"]
        ).to_string(index=False)
    )

    print()
    print("Wrote:", confirmatory_omni_path)
    print("Wrote:", null_omni_path)
    print("Wrote:", confirmatory_pair_path)
    print("Wrote:", sensitivity_path)
    print("Wrote:", readable_path)
    print("Wrote:", variant_path)
    print("Wrote:", report_path)


if __name__ == "__main__":
    main()
