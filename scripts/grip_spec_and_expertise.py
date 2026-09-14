"""
SIMETRIC fix pipeline — step 8: SPECIFICATION COMPARISON AND EXPERTISE TESTS
============================================================================
Two jobs:

  A. Compare the nominal (multinomial) GEE against the baseline-category
     binomial decomposition on grounds that do NOT depend on which produces
     more significant contrasts.
  B. Run the omnibus expertise test the manuscript reports as null (P = 0.354),
     plus per-category expertise contrasts with BH correction.

Note on specification choice
----------------------------
The two approaches estimate the same target but use different information. The
nominal GEE fits all baseline-category logits jointly on all 414 attempts and
models the association between categories. The binomial decomposition fits each
contrast on the SUBSET containing that category and IPG only, discarding the
remaining attempts, and treats the contrasts as if they were separate studies.
Selecting between them on the basis of how many contrasts reach significance is
outcome-dependent model selection. The pre-specified model in the manuscript is
the multinomial GEE, and it should be retained unless it fails diagnostically.

Usage
-----
    python simetric_08_grip_spec_and_expertise.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.genmod.cov_struct import Exchangeable, GlobalOddsRatio

ROOT = Path(r"<SET_YOUR_ANALYSIS_ROOT>")
OUT = ROOT / "corrected"
OUT.mkdir(exist_ok=True)

FREEZE = (ROOT / "simetric_hand_recovery_output" / "SIMETRIC_grip_final_freeze_v1_8_2"
          / "02_SIMETRIC_grip_analysis_resolved_414_FINAL_v1_8_2.csv")
EYE = (ROOT / "SIMETRIC-eye-tracking-repository" / "data" / "intermediate"
       / "attempt_intervals_cleaned.csv")

DEVICE_FAMILY_MAP = {"DEVICE1": "MST", "DEVICE2": "ATG", "DEVICE3": "CON"}
REJECTED_OVERRIDES = {"P6_T11": "MST", "P19_T03": "MST"}
GRIPS = ["EPG", "TC", "OTHER", "IPG"]
CODE_TO_GRIP = {0.0: "EPG", 1.0: "TC", 2.0: "OTHER"}
DEVICES = ["MST", "ATG", "CON"]
EXPERTISE = ["Novice", "Intermediate", "Expert"]
FORMULA = ("grip_code ~ C(device, Treatment('MST')) "
           "+ C(expertise, Treatment('Novice')) + repetition_c")


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
    df = df.rename(columns={"Expertise": "expertise"}).dropna(
        subset=["device", "grip", "repetition", "expertise"])
    df["device"] = pd.Categorical(df["device"], categories=DEVICES)
    df["expertise"] = pd.Categorical(df["expertise"], categories=EXPERTISE)
    df["repetition_c"] = df["repetition"] - df["repetition"].mean()
    df["grip_code"] = df["grip"].map({g: i for i, g in enumerate(GRIPS)})
    return df


def wald(res, needle):
    terms = [t for t in res.params.index if needle in t]
    if not terms:
        return None
    rmat = np.array([[1.0 if p == t else 0.0 for p in res.params.index] for t in terms])
    try:
        w = res.wald_test(rmat, scalar=False)
    except TypeError:
        w = res.wald_test(rmat)
    chi2 = float(np.squeeze(np.asarray(w.statistic)))
    return chi2, len(terms), stats.chi2.sf(chi2, len(terms)), terms


def main():
    df = load()
    rule("DATA")
    print(f"  attempts {len(df)}   participants {df['participant'].nunique()}")
    print("  expertise strata (participants): " + ", ".join(
        f"{e}={df[df['expertise'] == e]['participant'].nunique()}" for e in EXPERTISE))
    print("  expertise strata (attempts):     " + ", ".join(
        f"{e}={int((df['expertise'] == e).sum())}" for e in EXPERTISE))

    gid = df["participant"].astype("category").cat.codes.to_numpy()
    res = smf.nominal_gee(FORMULA, groups=gid, data=df,
                          cov_struct=GlobalOddsRatio("nominal")).fit(maxiter=100)

    # ---------------------------------------------------------------- A
    rule("A. SPECIFICATION COMPARISON — on grounds independent of significance")

    print("\n  1. Information used per contrast")
    print("     nominal GEE: every contrast is estimated from all 414 attempts,")
    print("     jointly, with the between-category association modelled.")
    print("     binomial decomposition: each contrast uses only its own subset -")
    for cat in ["EPG", "TC", "OTHER"]:
        n = int(df["grip"].isin([cat, "IPG"]).sum())
        print(f"       {cat:<5} vs IPG uses {n} of {len(df)} attempts "
              f"({n / len(df):.0%}); {len(df) - n} discarded")
    print("     Discarded attempts carry information about the reference category,")
    print("     so the decomposition is less efficient and its standard errors are")
    print("     not directly comparable across contrasts.")

    print("\n  2. Multiplicity footing")
    print("     The six nominal-GEE contrasts come from one fitted model, so BH")
    print("     across them is a coherent family. The six binomial contrasts come")
    print("     from three models fitted to overlapping subsets that all share the")
    print("     IPG attempts, so they are neither independent nor a clean family.")

    print("\n  3. Convergence and diagnostics")
    print(f"     nominal GEE converged in {res.model.cov_struct.__class__.__name__} "
          f"structure; observations {int(res.nobs)}, clusters {df['participant'].nunique()}")
    se = res.bse
    print(f"     largest standard error {se.max():.3f} on {se.idxmax()}")
    print(f"     any non-finite parameters: {bool(~np.isfinite(res.params).all())}")

    print("\n  CONCLUSION: the nominal GEE is the appropriate specification. It is")
    print("  the model pre-specified in the manuscript, it uses all attempts for")
    print("  every contrast, and its contrast family is coherent for BH. The")
    print("  binomial decomposition should be reported as a cross-check only.")
    print("  Choosing between them by counting significant contrasts would be")
    print("  outcome-dependent selection and is not defensible.")

    # ---------------------------------------------------------------- B
    rule("B. OMNIBUS TESTS FROM THE NOMINAL GEE")
    for name, needle, manuscript in [
        ("configuration", "device", "chi2(6)=26.86, P=0.00015"),
        ("expertise", "expertise", "P=0.354"),
        ("repetition", "repetition_c", "P=0.365"),
    ]:
        got = wald(res, needle)
        if got is None:
            print(f"  {name}: no matching terms")
            continue
        chi2, dfree, p, _ = got
        print(f"  {name:<14} Wald chi2({dfree}) = {chi2:7.2f}   P = {p:.5g}"
              f"    [manuscript: {manuscript}]")

    rule("EXPERTISE CONTRASTS BY GRIP CATEGORY (reference Novice, IPG)")
    rows = []
    for term in [t for t in res.params.index if "expertise" in t]:
        cat_code = float(term.split("[")[-1].rstrip("]"))
        level = term.split("[T.")[1].split("]")[0]
        est, s = res.params[term], res.bse[term]
        z = est / s
        rows.append({
            "grip": CODE_TO_GRIP.get(cat_code, str(cat_code)),
            "contrast": f"{level} vs Novice",
            "RRR": np.exp(est), "lo": np.exp(est - 1.96 * s), "hi": np.exp(est + 1.96 * s),
            "p_raw": 2 * stats.norm.sf(abs(z)),
        })
    tab = pd.DataFrame(rows)
    tab["q_BH"] = bh(tab["p_raw"])
    tab["reported"] = tab.apply(
        lambda r: f"RRR {r.RRR:.2f} ({r.lo:.2f}\u2013{r.hi:.2f}); p={r.p_raw:.3f}; q={r.q_BH:.3f}",
        axis=1)
    print(tab[["grip", "contrast", "reported"]].to_string(index=False))
    tab.to_csv(OUT / "grip_expertise_contrasts_v1_8_2.csv", index=False)

    rule("INTERPRETATION")
    exp_test = wald(res, "expertise")
    if exp_test and exp_test[2] < 0.05:
        print("  The omnibus expertise effect IS significant on this dataset, which")
        print("  contradicts the null expertise result reported in the manuscript.")
    else:
        print("  The omnibus expertise effect is NOT significant, so individual")
        print("  expertise contrasts should not be reported as findings even where")
        print("  a single coefficient reaches nominal significance.")
    n_expert = df[df['expertise'] == 'Expert']['participant'].nunique()
    print(f"\n  Either way, the Expert stratum contains {n_expert} clinicians. Expertise is a")
    print("  participant-level factor, so its effective sample size is the number of")
    print("  clinicians, not the number of attempts. Any expertise finding here is")
    print("  exploratory and should be labelled as such.")
    print(f"\n  wrote {OUT / 'grip_expertise_contrasts_v1_8_2.csv'}")


if __name__ == "__main__":
    main()
