from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Independence
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2
from patsy import build_design_matrices


PAIRWISE = [
    ("MST_vs_CON", "MST", "CON"),
    ("ATG_vs_CON", "ATG", "CON"),
    ("MST_vs_ATG", "MST", "ATG"),
]


def as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def wald_from_indices(result, indices):
    indices = list(indices)
    if not indices:
        return np.nan, 0, np.nan
    b = result.params.to_numpy(float)[indices]
    cov = np.asarray(result.cov_params(), dtype=float)[np.ix_(indices, indices)]
    stat = float(b.T @ np.linalg.pinv(cov) @ b)
    return stat, len(indices), float(chi2.sf(stat, len(indices)))


def interaction_names(names, level):
    out = []
    for n in names:
        s = str(n)
        if f"[T.{level}]" in s and "bs(t_norm" in s and ":" in s:
            out.append(s)
    return out


def pairwise_interaction_test(result, high, low):
    names = list(result.params.index)
    hi = interaction_names(names, high) if high != "CON" else []
    lo = interaction_names(names, low) if low != "CON" else []

    # Match spline basis terms by suffix.
    def basis_key(name):
        return name.split("bs(t_norm", 1)[1] if "bs(t_norm" in name else name

    hi_map = {basis_key(x): x for x in hi}
    lo_map = {basis_key(x): x for x in lo}
    keys = sorted(set(hi_map) | set(lo_map))
    if not keys:
        return np.nan, 0, np.nan

    R = np.zeros((len(keys), len(names)))
    for r, key in enumerate(keys):
        if key in hi_map:
            R[r, names.index(hi_map[key])] += 1
        if key in lo_map:
            R[r, names.index(lo_map[key])] -= 1

    beta = result.params.to_numpy(float)
    cov = np.asarray(result.cov_params(), dtype=float)
    rb = R @ beta
    rcov = R @ cov @ R.T
    stat = float(rb.T @ np.linalg.pinv(rcov) @ rb)
    return stat, len(keys), float(chi2.sf(stat, len(keys)))


def fit_one(data: pd.DataFrame, coord: str):
    d = data.copy()
    d["_y"] = pd.to_numeric(d[coord], errors="coerce")
    d["t_norm"] = pd.to_numeric(d["normalized_time_0_1"], errors="coerce")
    d["device_repetition_c"] = pd.to_numeric(d["device_repetition_c"], errors="coerce")
    d["chronological_attempt_sequence_c"] = pd.to_numeric(
        d["chronological_attempt_sequence_c"], errors="coerce"
    )
    d = d.dropna(
        subset=[
            "_y", "t_norm", "participant", "device", "experience_stratum",
            "device_repetition_c", "chronological_attempt_sequence_c",
        ]
    )

    formula = (
        '_y ~ C(device, Treatment(reference="CON"))'
        ' * bs(t_norm, df=6, degree=3, include_intercept=False)'
        ' + C(experience_stratum, Treatment(reference="None"))'
        ' + device_repetition_c'
        ' + chronological_attempt_sequence_c'
    )

    model = smf.gee(
        formula=formula,
        groups="participant",
        data=d,
        family=sm.families.Gaussian(),
        cov_struct=Independence(),
    )
    try:
        result = model.fit(cov_type="bias_reduced", maxiter=200)
        cov_type = "bias_reduced"
    except Exception:
        result = model.fit(cov_type="robust", maxiter=200)
        cov_type = "robust_fallback"

    names = list(result.params.index)
    ix = [
        i for i, n in enumerate(names)
        if "C(device" in str(n) and "bs(t_norm" in str(n) and ":" in str(n)
    ]
    stat, df_test, p = wald_from_indices(result, ix)
    return result, d, cov_type, stat, df_test, p


def prediction_grid(result):
    rows = []
    design_info = result.model.data.design_info
    beta = result.params.to_numpy(float)
    cov = np.asarray(result.cov_params(), dtype=float)

    for device in ["CON", "ATG", "MST"]:
        grid = pd.DataFrame({
            "device": [device] * 101,
            "experience_stratum": ["None"] * 101,
            "device_repetition_c": [0.0] * 101,
            "chronological_attempt_sequence_c": [0.0] * 101,
            "t_norm": np.linspace(0, 1, 101),
        })
        X = np.asarray(build_design_matrices([design_info], grid)[0], dtype=float)
        pred = X @ beta
        var = np.einsum("ij,jk,ik->i", X, cov, X)
        se = np.sqrt(np.maximum(var, 0))
        for t, y, s in zip(grid["t_norm"], pred, se):
            rows.append({
                "device": device,
                "normalized_time_0_1": t,
                "adjusted_centered_position": y,
                "ci95_low": y - 1.95996398454 * s,
                "ci95_high": y + 1.95996398454 * s,
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    traj = pd.read_csv(args.trajectory, keep_default_na=False)
    master = pd.read_csv(
        args.master, dtype=str, keep_default_na=False, na_filter=False
    )

    keep = [
        "attempt_id", "hand", "participant", "device", "experience_stratum",
        "device_repetition_c", "chronological_attempt_sequence_c",
        "analysis_main", "analysis_highqc_sensitivity",
    ]
    m = master[keep].drop_duplicates(["attempt_id", "hand"])
    df = traj.merge(m, on=["attempt_id", "hand"], how="inner", validate="many_to_one")

    if "tracking_valid" in df.columns:
        df["_valid"] = as_bool(df["tracking_valid"])
    else:
        df["_valid"] = (
            pd.to_numeric(df["x_norm"], errors="coerce").notna()
            & pd.to_numeric(df["y_norm"], errors="coerce").notna()
        )

    df["x_norm"] = pd.to_numeric(df["x_norm"], errors="coerce")
    df["y_norm"] = pd.to_numeric(df["y_norm"], errors="coerce")
    df["normalized_time_0_1"] = pd.to_numeric(
        df["normalized_time_0_1"], errors="coerce"
    )

    # Main inferential trajectory = within-attempt centred shape.
    # This removes arbitrary camera framing without relying on a possibly-missing t=0 point.
    centers = (
        df[df["_valid"]]
        .groupby(["attempt_id", "hand"])[["x_norm", "y_norm"]]
        .mean()
        .rename(columns={"x_norm": "x_center", "y_norm": "y_center"})
        .reset_index()
    )
    df = df.merge(centers, on=["attempt_id", "hand"], how="left")
    df["x_centered"] = df["x_norm"] - df["x_center"]
    df["y_centered"] = df["y_norm"] - df["y_center"]

    coverage = (
        df.groupby(["attempt_id", "hand"])["_valid"]
        .mean()
        .rename("trajectory_valid_fraction")
        .reset_index()
    )
    df = df.merge(coverage, on=["attempt_id", "hand"], how="left")

    set_defs = {
        "main_gopro_ge60": ("analysis_main", 0.60),
        "highqc_gopro_ge80": ("analysis_highqc_sensitivity", 0.80),
    }

    omnibus_rows = []
    pairwise_rows = []
    prediction_rows = []
    status_rows = []

    for set_name, (flag, min_valid) in set_defs.items():
        base = df[
            as_bool(df[flag])
            & df["_valid"]
            & df["trajectory_valid_fraction"].ge(min_valid)
        ].copy()

        for hand in ["left", "right"]:
            hd = base[base["hand"].eq(hand)].copy()
            for coord in ["x_centered", "y_centered"]:
                try:
                    result, fitdata, cov_type, stat, dft, p = fit_one(hd, coord)
                    status_rows.append({
                        "analysis_set": set_name,
                        "hand": hand,
                        "coordinate": coord,
                        "rows": len(fitdata),
                        "attempts": fitdata["attempt_id"].nunique(),
                        "participants": fitdata["participant"].nunique(),
                        "fit_status": "PASS",
                        "cov_type": cov_type,
                        "error": "",
                    })
                    omnibus_rows.append({
                        "analysis_set": set_name,
                        "hand": hand,
                        "coordinate": coord,
                        "wald_chi2_device_by_shape": stat,
                        "df": dft,
                        "p_raw": p,
                        "p_holm_across_main_4_tests": np.nan,
                    })

                    for label, high, low in PAIRWISE:
                        ps, pdf, pp = pairwise_interaction_test(result, high, low)
                        pairwise_rows.append({
                            "analysis_set": set_name,
                            "hand": hand,
                            "coordinate": coord,
                            "contrast": label,
                            "wald_chi2_shape": ps,
                            "df": pdf,
                            "p_raw": pp,
                            "p_holm_within_coordinate": np.nan,
                        })

                    pred = prediction_grid(result)
                    pred.insert(0, "coordinate", coord)
                    pred.insert(0, "hand", hand)
                    pred.insert(0, "analysis_set", set_name)
                    prediction_rows.append(pred)

                except Exception as e:
                    status_rows.append({
                        "analysis_set": set_name,
                        "hand": hand,
                        "coordinate": coord,
                        "rows": len(hd),
                        "attempts": hd["attempt_id"].nunique(),
                        "participants": hd["participant"].nunique(),
                        "fit_status": "FAIL",
                        "cov_type": "",
                        "error": repr(e),
                    })

    omni = pd.DataFrame(omnibus_rows)
    pair = pd.DataFrame(pairwise_rows)
    preds = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    status = pd.DataFrame(status_rows)

    # Main 4-test Holm correction: left/right × X/Y.
    if len(omni):
        mask = omni["analysis_set"].eq("main_gopro_ge60") & omni["p_raw"].notna()
        if mask.any():
            _, adj, _, _ = multipletests(
                omni.loc[mask, "p_raw"].astype(float), method="holm"
            )
            omni.loc[mask, "p_holm_across_main_4_tests"] = adj

    if len(pair):
        for _, idx in pair.groupby(
            ["analysis_set", "hand", "coordinate"]
        ).groups.items():
            idx = list(idx)
            p = pd.to_numeric(pair.loc[idx, "p_raw"], errors="coerce")
            valid = p.notna()
            if valid.any():
                _, adj, _, _ = multipletests(p[valid], method="holm")
                pair.loc[p[valid].index, "p_holm_within_coordinate"] = adj

    # 2-D adjusted curve separation, descriptive.
    sep_rows = []
    if len(preds):
        mainp = preds[preds["analysis_set"].eq("main_gopro_ge60")]
        for hand in ["left", "right"]:
            x = mainp[(mainp["hand"] == hand) & (mainp["coordinate"] == "x_centered")]
            y = mainp[(mainp["hand"] == hand) & (mainp["coordinate"] == "y_centered")]
            for label, high, low in PAIRWISE:
                xh = x[x["device"] == high].sort_values("normalized_time_0_1")
                xl = x[x["device"] == low].sort_values("normalized_time_0_1")
                yh = y[y["device"] == high].sort_values("normalized_time_0_1")
                yl = y[y["device"] == low].sort_values("normalized_time_0_1")
                if not (len(xh) == len(xl) == len(yh) == len(yl) == 101):
                    continue
                dx = xh["adjusted_centered_position"].to_numpy() - xl["adjusted_centered_position"].to_numpy()
                dy = yh["adjusted_centered_position"].to_numpy() - yl["adjusted_centered_position"].to_numpy()
                sep_rows.append({
                    "hand": hand,
                    "contrast": label,
                    "rms_2d_separation_norm": float(np.sqrt(np.mean(dx*dx + dy*dy))),
                    "max_2d_separation_norm": float(np.max(np.sqrt(dx*dx + dy*dy))),
                })
    sep = pd.DataFrame(sep_rows)

    status.to_csv(out / "trajectory_model_status.csv", index=False)
    omni.to_csv(out / "trajectory_shape_omnibus.csv", index=False)
    pair.to_csv(out / "trajectory_shape_pairwise.csv", index=False)
    preds.to_csv(out / "trajectory_adjusted_curves.csv", index=False)
    sep.to_csv(out / "trajectory_2d_separation.csv", index=False)
    coverage.to_csv(out / "trajectory_attempt_valid_fraction.csv", index=False)

    # Figures.
    if len(preds):
        mainp = preds[preds["analysis_set"].eq("main_gopro_ge60")]
        for hand in ["left", "right"]:
            for coord, ylabel in [
                ("x_centered", "Centred normalized X"),
                ("y_centered", "Centred normalized Y"),
            ]:
                fig, ax = plt.subplots(figsize=(8, 5))
                q = mainp[(mainp["hand"] == hand) & (mainp["coordinate"] == coord)]
                for dev, g in q.groupby("device"):
                    g = g.sort_values("normalized_time_0_1")
                    ax.plot(g["normalized_time_0_1"], g["adjusted_centered_position"], label=dev)
                    ax.fill_between(
                        g["normalized_time_0_1"],
                        g["ci95_low"],
                        g["ci95_high"],
                        alpha=0.15,
                    )
                ax.set_xlabel("Normalized attempt time")
                ax.set_ylabel(ylabel)
                ax.set_title(f"{hand.capitalize()} hand adjusted {coord[0].upper()} trajectory")
                ax.legend()
                fig.tight_layout()
                fig.savefig(out / f"trajectory_{hand}_{coord}_main.png", dpi=180)
                plt.close(fig)

            # 2D curve.
            fig, ax = plt.subplots(figsize=(6, 6))
            x = mainp[(mainp["hand"] == hand) & (mainp["coordinate"] == "x_centered")]
            y = mainp[(mainp["hand"] == hand) & (mainp["coordinate"] == "y_centered")]
            for dev in ["CON", "ATG", "MST"]:
                gx = x[x["device"] == dev].sort_values("normalized_time_0_1")
                gy = y[y["device"] == dev].sort_values("normalized_time_0_1")
                if len(gx) == 101 and len(gy) == 101:
                    ax.plot(
                        gx["adjusted_centered_position"],
                        gy["adjusted_centered_position"],
                        label=dev,
                    )
                    ax.scatter(
                        gx["adjusted_centered_position"].iloc[0],
                        gy["adjusted_centered_position"].iloc[0],
                        marker="o",
                    )
                ax.set_xlabel("Centred normalized X")
                ax.set_ylabel("Centred normalized Y")
                ax.set_title(f"{hand.capitalize()} hand adjusted 2-D trajectory")
                ax.legend()
            fig.tight_layout()
            fig.savefig(out / f"trajectory_{hand}_2d_main.png", dpi=180)
            plt.close(fig)

    print("=== LPC X/Y TRAJECTORY SHAPE ANALYSIS COMPLETE ===")
    print()
    print("Model status:")
    print(status.groupby(["analysis_set", "fit_status"]).size().reset_index(name="models").to_string(index=False))
    print()
    print("Main trajectory-shape omnibus tests:")
    q = omni[omni["analysis_set"].eq("main_gopro_ge60")]
    print(q.to_string(index=False))
    print()
    print("Main pairwise shape tests:")
    print(pair[pair["analysis_set"].eq("main_gopro_ge60")].to_string(index=False))
    print()
    print("2-D adjusted curve separation:")
    print(sep.to_string(index=False) if len(sep) else "<none>")
    print()
    print("Inference is on within-attempt centred trajectory shape; absolute camera-frame position is not interpreted.")


if __name__ == "__main__":
    main()
