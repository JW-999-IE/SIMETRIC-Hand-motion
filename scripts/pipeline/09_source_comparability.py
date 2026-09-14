from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ensure_dir


DEFAULT_METRICS = [
    "straight_line_displacement_3d",
    "path_length_3d",
    "mean_velocity",
    "peak_velocity",
    "path_efficiency_3d",
    "mean_abs_jerk",
    "index_tip_displacement_3d",
    "thumb_tip_displacement_3d",
]


def classify_metric(n: int, corr: float, median_rel_diff: float) -> str:
    if n < 10:
        return "INSUFFICIENT EVIDENCE"
    if np.isfinite(corr) and corr >= 0.80 and np.isfinite(median_rel_diff) and median_rel_diff <= 0.20:
        return "COMPARABLE"
    return "SOURCE-SENSITIVE"


def main():
    parser = argparse.ArgumentParser(description="Compare CAE and GoPro motion metrics.")
    parser.add_argument("--gopro", required=True)
    parser.add_argument("--cae", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    args = parser.parse_args()

    g = pd.read_csv(args.gopro)
    c = pd.read_csv(args.cae)

    keys = ["attempt_id", "participant", "hand"]
    merged = g.merge(c, on=keys, suffixes=("_gopro", "_cae"))
    out_root = ensure_dir(Path(args.output_root))
    paired_path = out_root / "paired_source_comparison.csv"
    merged.to_csv(paired_path, index=False)

    rows = []
    for hand in sorted(merged["hand"].dropna().unique()):
        h = merged[merged["hand"].eq(hand)]

        for metric in args.metrics:
            a_col = f"{metric}_gopro"
            b_col = f"{metric}_cae"
            if a_col not in h.columns or b_col not in h.columns:
                continue

            a = pd.to_numeric(h[a_col], errors="coerce")
            b = pd.to_numeric(h[b_col], errors="coerce")
            valid = a.notna() & b.notna()
            a = a[valid]
            b = b[valid]
            n = len(a)
            if n == 0:
                continue

            diff = b - a
            denom = a.abs().replace(0, np.nan)
            rel = (diff.abs() / denom).replace([np.inf, -np.inf], np.nan)
            corr = float(a.corr(b)) if n >= 2 else np.nan
            median_rel = float(rel.median()) if rel.notna().any() else np.nan

            rows.append({
                "hand": hand,
                "metric": metric,
                "n_pairs": n,
                "correlation": corr,
                "mean_difference_cae_minus_gopro": float(diff.mean()),
                "median_abs_relative_difference": median_rel,
                "classification": classify_metric(n, corr, median_rel),
            })

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(a, b)
            lo = float(np.nanmin([a.min(), b.min()]))
            hi = float(np.nanmax([a.max(), b.max()]))
            ax.plot([lo, hi], [lo, hi])
            ax.set_xlabel("GoPro")
            ax.set_ylabel("CAE-HAND")
            ax.set_title(f"{hand}: {metric}")
            fig.tight_layout()
            fig.savefig(out_root / f"{hand}_{metric}_agreement.png", dpi=160)
            plt.close(fig)

    summary = pd.DataFrame(rows)
    summary_path = out_root / "source_comparability_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Wrote: {paired_path}")
    print(f"Wrote: {summary_path}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
