from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OUTCOMES = [
    ("mean_accel_norm_s2", "continuous_positive"),
    ("rms_accel_norm_s2", "continuous_positive"),
    ("peak_accel_norm_s2", "continuous_positive"),
    ("sparc", "continuous_signed"),
    ("log_dimensionless_jerk", "continuous_signed"),
    ("velocity_peak_count", "count"),
    ("path_length_norm", "continuous_positive"),
    ("path_efficiency", "bounded_0_1"),
    ("stillness_fraction", "bounded_0_1"),
    ("first_sustained_hand_movement_delay_sec", "continuous_nonnegative"),
]


def to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def safe_skew(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) < 3:
        return np.nan
    return float(x.skew())


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Pre-model distribution and estimability audit for the frozen SIMETRIC "
            "hand-motion analysis population. Does not fit device-effect models or inspect p-values."
        )
    )
    ap.add_argument("--master", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.master, dtype=str).fillna("")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    required = {
        "analysis_main",
        "analysis_highqc_sensitivity",
        "analysis_recovery_sensitivity",
        "participant",
        "hand",
        "device",
        "experience_stratum",
        "device_repetition_c",
        "chronological_attempt_sequence_c",
        "motion_source",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            "Frozen master is missing required columns: "
            + ", ".join(sorted(missing))
        )

    main_mask = to_bool(df["analysis_main"])
    main = df.loc[main_mask].copy()

    # ---------------------------
    # 1. Outcome distribution audit
    # ---------------------------
    rows = []

    for hand in ["left", "right"]:
        hdf = main[main["hand"].eq(hand)].copy()

        for outcome, kind in OUTCOMES:
            if outcome not in hdf.columns:
                continue

            x = pd.to_numeric(hdf[outcome], errors="coerce")
            valid = x.dropna()

            zero_n = int((valid == 0).sum())
            neg_n = int((valid < 0).sum())
            one_n = int((valid == 1).sum())

            if len(valid):
                q01 = float(valid.quantile(.01))
                q25 = float(valid.quantile(.25))
                med = float(valid.median())
                q75 = float(valid.quantile(.75))
                q99 = float(valid.quantile(.99))
                vmin = float(valid.min())
                vmax = float(valid.max())
                mean = float(valid.mean())
                sd = float(valid.std(ddof=1)) if len(valid) > 1 else np.nan
            else:
                q01 = q25 = med = q75 = q99 = vmin = vmax = mean = sd = np.nan

            skew = safe_skew(valid)

            # Pre-model family/transform suggestion based on scale + shape only.
            if kind in {"continuous_positive", "continuous_nonnegative"}:
                if len(valid) and neg_n == 0 and np.isfinite(skew) and skew > 1.0:
                    if zero_n:
                        recommendation = "Gaussian GEE on log1p(outcome) sensitivity to raw-scale model"
                    else:
                        recommendation = "Gaussian GEE on log(outcome) sensitivity to raw-scale model"
                else:
                    recommendation = "Gaussian GEE on raw outcome"
            elif kind == "count":
                if len(valid) and mean > 0 and np.isfinite(sd) and sd**2 > mean * 1.5:
                    recommendation = "Negative-binomial GEE preferred; Poisson GEE as sensitivity"
                else:
                    recommendation = "Poisson GEE; inspect overdispersion"
            elif kind == "bounded_0_1":
                boundary = zero_n + one_n
                if boundary:
                    recommendation = (
                        "Gaussian GEE on raw proportion with robust SE; "
                        "logit-transformed sensitivity only for non-boundary rows"
                    )
                else:
                    recommendation = (
                        "Gaussian GEE on raw proportion with robust SE; "
                        "logit-transformed sensitivity"
                    )
            else:
                recommendation = "Gaussian GEE on raw outcome"

            rows.append({
                "hand": hand,
                "outcome": outcome,
                "outcome_kind": kind,
                "rows_in_hand_main_set": len(hdf),
                "nonmissing_n": len(valid),
                "participants": hdf.loc[x.notna(), "participant"].nunique(),
                "device_levels": hdf.loc[x.notna(), "device"].nunique(),
                "zero_n": zero_n,
                "negative_n": neg_n,
                "one_n": one_n,
                "mean": mean,
                "sd": sd,
                "skew": skew,
                "min": vmin,
                "q01": q01,
                "q25": q25,
                "median": med,
                "q75": q75,
                "q99": q99,
                "max": vmax,
                "pre_model_recommendation": recommendation,
            })

    dist = pd.DataFrame(rows)
    dist_path = outdir / "scalar_outcome_distribution_audit.csv"
    dist.to_csv(dist_path, index=False)

    # ---------------------------
    # 2. Estimability by hand
    # ---------------------------
    estimability_rows = []

    for hand in ["left", "right"]:
        h = main[main["hand"].eq(hand)].copy()

        # Participant-level device coverage.
        pdev = (
            h.groupby("participant")["device"]
            .nunique()
            .rename("device_levels")
            .reset_index()
        )

        participants_all3 = int((pdev["device_levels"] == 3).sum())
        participants_2plus = int((pdev["device_levels"] >= 2).sum())
        participants_1 = int((pdev["device_levels"] == 1).sum())

        row = {
            "hand": hand,
            "rows": len(h),
            "participants": h["participant"].nunique(),
            "participant_device_groups": len(
                h[["participant", "device"]].drop_duplicates()
            ),
            "participants_with_all_3_devices": participants_all3,
            "participants_with_at_least_2_devices": participants_2plus,
            "participants_with_only_1_device": participants_1,
            "MST_rows": int((h["device"] == "MST").sum()),
            "ATG_rows": int((h["device"] == "ATG").sum()),
            "CON_rows": int((h["device"] == "CON").sum()),
            "Higher_rows": int((h["experience_stratum"] == "Higher").sum()),
            "Some_rows": int((h["experience_stratum"] == "Some").sum()),
            "None_rows": int((h["experience_stratum"] == "None").sum()),
        }
        estimability_rows.append(row)

    estimability = pd.DataFrame(estimability_rows)
    estimability_path = outdir / "model_estimability_by_hand.csv"
    estimability.to_csv(estimability_path, index=False)

    # ---------------------------
    # 3. Device × experience balance
    # ---------------------------
    balance = (
        main.groupby(["hand", "device", "experience_stratum"], dropna=False)
        .agg(
            rows=("attempt_id", "size"),
            participants=("participant", "nunique"),
        )
        .reset_index()
    )
    balance_path = outdir / "device_experience_balance_main.csv"
    balance.to_csv(balance_path, index=False)

    # ---------------------------
    # 4. Participant-level main-set coverage
    # ---------------------------
    participant = (
        main.groupby(["participant", "hand"], dropna=False)
        .agg(
            rows=("attempt_id", "size"),
            devices=("device", "nunique"),
            MST_rows=("device", lambda s: int((s == "MST").sum())),
            ATG_rows=("device", lambda s: int((s == "ATG").sum())),
            CON_rows=("device", lambda s: int((s == "CON").sum())),
        )
        .reset_index()
    )
    participant_path = outdir / "participant_hand_device_coverage_main.csv"
    participant.to_csv(participant_path, index=False)

    # ---------------------------
    # Console report
    # ---------------------------
    print("=== PRE-MODEL DISTRIBUTION / ESTIMABILITY AUDIT ===")
    print("Frozen main-analysis rows:", len(main))
    print()

    print("=== ESTIMABILITY BY HAND ===")
    print(estimability.to_string(index=False))
    print()

    print("=== OUTCOME DISTRIBUTIONS / PRE-MODEL RECOMMENDATIONS ===")
    print(
        dist[
            [
                "hand",
                "outcome",
                "nonmissing_n",
                "participants",
                "zero_n",
                "negative_n",
                "skew",
                "median",
                "q99",
                "max",
                "pre_model_recommendation",
            ]
        ].to_string(index=False)
    )
    print()

    print("=== DEVICE × EXPERIENCE BALANCE ===")
    print(balance.to_string(index=False))
    print()

    print("Wrote:", dist_path)
    print("Wrote:", estimability_path)
    print("Wrote:", balance_path)
    print("Wrote:", participant_path)
    print()
    print(
        "No device-effect model has been fitted. Model families/transforms above "
        "are based only on outcome scale/distribution and are now suitable for "
        "pre-specifying the inferential models."
    )


if __name__ == "__main__":
    main()
