"""
SIMETRIC fix pipeline — step 7: REFIT THE GRIP MODEL ON FREEZE v1.8.2
=====================================================================
Refits the needle-entry grip posture model on the last verifiable dataset
(grip freeze v1.8.2, 414/414 resolved, hash-manifested) under the adjudicated
configuration identities, and reports everything the manuscript needs.

Specification, matching the manuscript
--------------------------------------
Participant-clustered multinomial (nominal) GEE. IPG is the reference grip
category, MST the reference configuration and Novice the reference expertise
stratum. Covariates: configuration, expertise, within-configuration repetition.
Benjamini-Hochberg correction across the six configuration-by-category contrasts.

Because multinomial GEE parameterisation varies between statsmodels versions,
the script also fits the equivalent set of baseline-category binomial GEEs
(each non-reference grip against IPG) as an independent cross-check. If the two
disagree materially, do not report either without investigating.

Sensitivity analyses reported: first five within-configuration repetitions, and
exclusion of the residual Other category.

Usage
-----
    python simetric_07_refit_grip.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.genmod.cov_struct import Exchangeable

ROOT = Path(r"<SET_YOUR_ANALYSIS_ROOT>")
OUT = ROOT / "corrected"
OUT.mkdir(exist_ok=True)

FREEZE = (ROOT / "simetric_hand_recovery_output" / "SIMETRIC_grip_final_freeze_v1_8_2"
          / "02_SIMETRIC_grip_analysis_resolved_414_FINAL_v1_8_2.csv")
EYE = (ROOT / "SIMETRIC-eye-tracking-repository" / "data" / "intermediate"
       / "attempt_intervals_cleaned.csv")

DEVICE_FAMILY_MAP = {"DEVICE1": "MST", "DEVICE2": "ATG", "DEVICE3": "CON"}
REJECTED_OVERRIDES = {"P6_T11": "MST", "P19_T03": "MST"}

GRIPS = ["EPG", "TC", "OTHER", "IPG"]          # IPG last => reference
GRIP_CODE = {g: i for i, g in enumerate(GRIPS)}
DEVICES = ["MST", "ATG", "CON"]                 # MST first => reference
EXPERTISE = ["Novice", "Intermediate", "Expert"]


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        adj[i] = prev
    return np.minimum(adj, 1.0)


def load():
    df = pd.read_csv(FREEZE, low_memory=False)
    df["attempt"] = df["final_attempt_id"].fillna(df["attempt_id"]).astype(str)
    df["device"] = df["device_family"].map(DEVICE_FAMILY_MAP)
    df["device"] = [REJECTED_OVERRIDES.get(a, d) for a, d in zip(df["attempt"], df["device"])]
    df["grip"] = df["final_grip"].astype(str).str.strip().str.upper()
    df.loc[~df["grip"].isin(GRIPS), "grip"] = "OTHER"
    df["repetition"] = pd.to_numeric(df["device_rep"], errors="coerce")

    eye = pd.read_csv(EYE, usecols=["Participant", "Expertise"]).drop_duplicates("Participant")
    df = df.merge(eye, left_on="participant", right_on="Participant", how="left")
    df = df.rename(columns={"Expertise": "expertise"})

    missing = df["expertise"].isna().sum()
    if missing:
        print(f"  !! {missing} attempts have no expertise stratum and will be dropped")
    df = df.dropna(subset=["device", "grip", "repetition", "expertise"])

    df["device"] = pd.Categorical(df["device"], categories=DEVICES)
    df["expertise"] = pd.Categorical(df["expertise"], categories=EXPERTISE)
    df["repetition_c"] = df["repetition"] - df["repetition"].mean()
    df["grip_code"] = df["grip"].map(GRIP_CODE)
    return df


def describe(df):
    t = pd.crosstab(df["grip"], df["device"]).reindex(index=GRIPS, columns=DEVICES).fillna(0).astype(int)
    t["Total"] = t.sum(axis=1)
    print("\nGrip x configuration")
    print(t.to_string())
    pct = (t[DEVICES] / t[DEVICES].sum(axis=0) * 100).round(1)
    print("\nColumn percentages")
    print(pct.to_string())
    t.to_csv(OUT / "grip_table_refit_v1_8_2.csv")
    return t


def baseline_logits(df, label="", cats=("EPG", "TC", "OTHER")):
    """Baseline-category logits: each non-reference grip vs IPG, clustered by participant.

    `cats` must exclude any category removed by a sensitivity analysis, otherwise
    the model is degenerate (all-zero outcome) and returns a meaningless RRR of
    1.00 with perfect separation. Benjamini-Hochberg is applied across exactly
    the contrasts actually fitted.
    """
    rows = []
    for cat in cats:
        sub = df[df["grip"].isin([cat, "IPG"])].copy()
        sub["y"] = (sub["grip"] == cat).astype(int)
        try:
            res = smf.gee(
                "y ~ C(device, Treatment('MST')) + C(expertise, Treatment('Novice')) + repetition_c",
                groups="participant", data=sub, family=sm.families.Binomial(),
                cov_struct=Exchangeable(),
            ).fit(cov_type="bias_reduced")
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {cat} vs IPG failed: {exc}")
            continue
        for dev in ["ATG", "CON"]:
            term = f"C(device, Treatment('MST'))[T.{dev}]"
            if term not in res.params.index:
                continue
            est, se = res.params[term], res.bse[term]
            z = est / se
            rows.append({
                "contrast": f"{cat} vs IPG, {dev} vs MST",
                "category": cat, "device": dev,
                "RRR": np.exp(est),
                "lo": np.exp(est - 1.96 * se),
                "hi": np.exp(est + 1.96 * se),
                "z": z, "p_raw": 2 * stats.norm.sf(abs(z)),
                "n": int(len(sub)),
            })
    tab = pd.DataFrame(rows)
    if tab.empty:
        return tab
    tab["q_BH"] = bh(tab["p_raw"])
    tab["reported"] = tab.apply(
        lambda r: f"RRR {r.RRR:.2f} ({r.lo:.2f}\u2013{r.hi:.2f}); "
                  f"q={'<0.001' if r.q_BH < 0.001 else f'{r.q_BH:.3f}'}", axis=1)
    print(f"\nBaseline-category contrasts{label}")
    print(tab[["contrast", "n", "reported", "p_raw"]].to_string(index=False))
    return tab


def nominal_gee(df):
    """Primary specification: participant-clustered nominal (multinomial) GEE."""
    try:
        from statsmodels.genmod.cov_struct import GlobalOddsRatio
    except Exception as exc:  # noqa: BLE001
        print(f"  !! GlobalOddsRatio unavailable ({exc}); skipping nominal GEE.")
        return None
    try:
        # groups must be numeric here; passing the participant string column
        # makes patsy try to coerce 'P1' to float.
        gid = df["participant"].astype("category").cat.codes.to_numpy()
        model = smf.nominal_gee(
            "grip_code ~ C(device, Treatment('MST')) + C(expertise, Treatment('Novice')) + repetition_c",
            groups=gid, data=df, cov_struct=GlobalOddsRatio("nominal"),
        )
        res = model.fit(maxiter=100)
    except Exception as exc:  # noqa: BLE001
        print(f"  !! nominal GEE failed to converge or is unsupported: {exc}")
        print("     Report the baseline-category results and say so in the Methods.")
        return None

    print(res.summary())
    dev_terms = [t for t in res.params.index if "device" in t]
    print(f"\n  configuration parameters ({len(dev_terms)}): {dev_terms}")
    if dev_terms:
        rmat = np.array([[1.0 if p == t else 0.0 for p in res.params.index] for t in dev_terms])
        try:
            w = res.wald_test(rmat, scalar=False)
        except TypeError:
            w = res.wald_test(rmat)
        chi2 = float(np.squeeze(np.asarray(w.statistic)))
        print(f"\n  omnibus configuration effect: Wald chi2({len(dev_terms)}) = {chi2:.2f}, "
              f"P = {stats.chi2.sf(chi2, len(dev_terms)):.5g}")
        print("  (manuscript reported chi2(6) = 26.86, P = 0.00015 on the lost dataset)")
    with open(OUT / "grip_nominal_gee_refit_v1_8_2.txt", "w", encoding="utf-8") as fh:
        fh.write(str(res.summary()))
    return res


def main():
    rule("STEP 7 — GRIP MODEL REFIT ON FREEZE v1.8.2")
    df = load()
    print(f"  attempts: {len(df)}   participants: {df['participant'].nunique()}")
    print("  configuration: " + ", ".join(
        f"{d}={int((df['device'] == d).sum())}" for d in DEVICES))
    print("  reference levels: grip IPG, configuration MST, expertise Novice")
    describe(df)

    rule("PRIMARY — participant-clustered nominal GEE")
    nominal_gee(df)

    rule("CROSS-CHECK — baseline-category binomial GEEs")
    main_tab = baseline_logits(df)
    if not main_tab.empty:
        main_tab.to_csv(OUT / "grip_contrasts_refit_v1_8_2.csv", index=False)
        print(f"\n  wrote {OUT / 'grip_contrasts_refit_v1_8_2.csv'}")

    rule("SENSITIVITY 1 — first five OBSERVED repetitions within each configuration")
    # device_rep is the scheduled repetition label and is already 1-5 for every
    # attempt including retries, so filtering on it removes nothing. The intended
    # restriction is on observed order within participant x configuration.
    ranked = df.sort_values("attempt_sequence").copy()
    ranked["_rank"] = ranked.groupby(["participant", "device"], observed=True).cumcount() + 1
    s1 = ranked[ranked["_rank"] <= 5]
    print(f"  attempts: {len(s1)}  (dropped {len(df) - len(s1)} later repetitions)")
    print("  configuration: " + ", ".join(
        f"{d}={int((s1['device'] == d).sum())}" for d in DEVICES))
    baseline_logits(s1, " (first 5 observed repetitions)")

    rule("SENSITIVITY 2 — excluding the residual Other category")
    s2 = df[df["grip"] != "OTHER"]
    print(f"  attempts: {len(s2)}")
    print("  Only EPG and TC are estimable here; the Other contrast is omitted")
    print("  rather than fitted on an empty outcome, and BH is applied across 4.")
    baseline_logits(s2, " (Other excluded)", cats=("EPG", "TC"))

    rule("INTERPRETATION NOTE")
    t = pd.crosstab(df["grip"], df["device"]).reindex(index=GRIPS, columns=DEVICES).fillna(0)
    for d in DEVICES:
        other, ipg = t.loc["OTHER", d], t.loc["IPG", d]
        print(f"  {d}: Other/IPG = {other:.0f}/{ipg:.0f} = {other / ipg:.3f}"
              if ipg else f"  {d}: no IPG")
    print("\n  On this dataset CON carries very few unclassifiable grips, so the")
    print("  largest Other-category contrast is ATG versus CON rather than ATG versus")
    print("  MST. Report whichever contrasts the model supports, and keep the grip")
    print("  endpoint exploratory given inter-rater agreement of 55.6% (kappa 0.344).")


if __name__ == "__main__":
    main()
