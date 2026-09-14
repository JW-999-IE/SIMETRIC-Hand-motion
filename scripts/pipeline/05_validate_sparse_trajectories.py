from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    canonicalize_columns,
    ensure_recovery_root,
    load_config,
    normalize_source,
    read_table,
    resolve_path,
    write_csv,
)


TRAJECTORY_ALIASES = {
    "attempt_id": ["Attempt_ID", "attempt id", "record_id"],
    "participant": ["Participant", "participant_id", "subject"],
    "source": ["source", "motion_source", "camera"],
    "target_hand": ["target_hand", "anatomical_hand", "hand"],
    "time_sec": ["time_sec", "video_time_sec", "attempt_time_sec", "timestamp_sec"],
    "wrist_x": ["wrist_x", "x_wrist", "x_WRIST", "x"],
    "wrist_y": ["wrist_y", "y_wrist", "y_WRIST", "y"],
}


def interpolate_trajectory(time: np.ndarray, x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(time)
    time, x, y = time[order], x[order], y[order]
    keep = np.isfinite(time) & np.isfinite(x) & np.isfinite(y)
    time, x, y = time[keep], x[keep], y[keep]
    unique_time, unique_indices = np.unique(time, return_index=True)
    x, y = x[unique_indices], y[unique_indices]
    if len(unique_time) < 4 or unique_time[-1] <= unique_time[0]:
        raise ValueError("Insufficient unique finite points")
    normalized = (unique_time - unique_time[0]) / (unique_time[-1] - unique_time[0])
    return np.column_stack([np.interp(grid, normalized, x), np.interp(grid, normalized, y)])


def path_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def maximum_gap_fraction(time: np.ndarray) -> float:
    if len(time) < 2 or time.max() <= time.min():
        return 1.0
    normalized = np.sort((time - time.min()) / (time.max() - time.min()))
    augmented = np.concatenate([[0.0], normalized, [1.0]])
    return float(np.diff(augmented).max())


def mask_indices(n: int, coverage: float, pattern: str, rng: np.random.Generator) -> np.ndarray:
    keep_n = max(4, int(round(n * coverage)))
    if keep_n >= n:
        return np.ones(n, dtype=bool)
    if pattern == "distributed":
        chosen = np.sort(rng.choice(n, size=keep_n, replace=False))
        mask = np.zeros(n, dtype=bool)
        mask[chosen] = True
        return mask
    if pattern == "single_gap":
        remove_n = n - keep_n
        start = int(rng.integers(0, n - remove_n + 1))
        mask = np.ones(n, dtype=bool)
        mask[start : start + remove_n] = False
        return mask
    raise ValueError(pattern)


def run_simulation(config: dict, trajectories: pd.DataFrame) -> pd.DataFrame:
    validation = config.get("sparse_validation") or {}
    coverage_levels = [float(value) for value in validation.get("coverage_levels", [0.35, 0.40, 0.50, 0.60, 0.70])]
    repetitions = int(validation.get("repetitions", 20))
    min_reference_points = int(validation.get("min_reference_points", 100))
    seed = int(validation.get("random_seed", 20260819))
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, 100)
    rows: list[dict] = []

    for key, group in trajectories.groupby(["attempt_id", "participant", "source", "target_hand"], dropna=False):
        group = group.sort_values("time_sec")
        time = pd.to_numeric(group["time_sec"], errors="coerce").to_numpy(float)
        x = pd.to_numeric(group["wrist_x"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(group["wrist_y"], errors="coerce").to_numpy(float)
        finite = np.isfinite(time) & np.isfinite(x) & np.isfinite(y)
        time, x, y = time[finite], x[finite], y[finite]
        if len(time) < min_reference_points or time.max(initial=0) <= time.min(initial=0):
            continue
        try:
            reference = interpolate_trajectory(time, x, y, grid)
        except ValueError:
            continue
        reference_path = path_length(reference)
        scale = max(
            float(np.hypot(np.ptp(reference[:, 0]), np.ptp(reference[:, 1]))),
            1e-9,
        )
        for coverage in coverage_levels:
            for pattern in ("distributed", "single_gap"):
                for repetition in range(repetitions):
                    mask = mask_indices(len(time), coverage, pattern, rng)
                    try:
                        reconstructed = interpolate_trajectory(time[mask], x[mask], y[mask], grid)
                    except ValueError:
                        continue
                    rmse = float(np.sqrt(np.mean(np.sum((reconstructed - reference) ** 2, axis=1))))
                    reconstructed_path = path_length(reconstructed)
                    rows.append(
                        {
                            "attempt_id": key[0],
                            "participant": key[1],
                            "source": key[2],
                            "target_hand": key[3],
                            "requested_coverage": coverage,
                            "realized_coverage": float(mask.mean()),
                            "pattern": pattern,
                            "repetition": repetition,
                            "max_gap_fraction": maximum_gap_fraction(time[mask]),
                            "normalized_trajectory_rmse": rmse / scale,
                            "relative_path_length_error": abs(reconstructed_path - reference_path) / max(reference_path, 1e-9),
                        }
                    )
    return pd.DataFrame(rows)


def summarize(config: dict, simulation: pd.DataFrame) -> pd.DataFrame:
    validation = config.get("sparse_validation") or {}
    rmse_tolerance = float(validation.get("p90_normalized_rmse_tolerance", 0.10))
    path_tolerance = float(validation.get("p90_path_error_tolerance", 0.15))
    if simulation.empty:
        return pd.DataFrame()
    summary = (
        simulation.groupby(["source", "target_hand", "pattern", "requested_coverage"], dropna=False)
        .agg(
            simulations=("attempt_id", "size"),
            reference_attempts=("attempt_id", "nunique"),
            median_max_gap=("max_gap_fraction", "median"),
            p90_max_gap=("max_gap_fraction", lambda x: x.quantile(0.90)),
            median_normalized_rmse=("normalized_trajectory_rmse", "median"),
            p90_normalized_rmse=("normalized_trajectory_rmse", lambda x: x.quantile(0.90)),
            median_path_error=("relative_path_length_error", "median"),
            p90_path_error=("relative_path_length_error", lambda x: x.quantile(0.90)),
        )
        .reset_index()
    )
    summary["validation_pass"] = (
        summary["p90_normalized_rmse"].le(rmse_tolerance)
        & summary["p90_path_error"].le(path_tolerance)
    )
    summary["decision_note"] = "Validation is source- and hand-specific; freeze the rule before outcome modelling."
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Outcome-blind masking validation for sparse hand trajectories."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--trajectories", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    recovery_root = ensure_recovery_root(config)
    path = (
        Path(args.trajectories).expanduser().resolve()
        if args.trajectories
        else resolve_path(config, config.get("trajectory_long_table"))
    )
    if path is None or not path.exists():
        raise FileNotFoundError("Provide --trajectories or config.trajectory_long_table")
    trajectories = canonicalize_columns(
        read_table(path),
        (config.get("column_maps") or {}).get("trajectories"),
        TRAJECTORY_ALIASES,
        required=["attempt_id", "participant", "source", "target_hand", "time_sec", "wrist_x", "wrist_y"],
    )
    trajectories["source"] = trajectories["source"].map(normalize_source)
    trajectories["target_hand"] = trajectories["target_hand"].astype(str).str.lower().str.strip()

    simulation = run_simulation(config, trajectories)
    summary = summarize(config, simulation)
    output_dir = recovery_root / "04_sparse_validation"
    write_csv(simulation, output_dir / "masking_simulations.csv")
    write_csv(summary, output_dir / "sparse_validation_summary.csv")
    print(f"Wrote outcome-blind sparse-trajectory validation to {output_dir}")


if __name__ == "__main__":
    main()
