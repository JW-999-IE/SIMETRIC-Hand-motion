from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import (
    ALLOWED_SOURCES,
    VIDEO_EXTENSIONS,
    ensure_recovery_root,
    ffprobe_metadata,
    infer_participant,
    load_config,
    natural_key,
    normalize_name,
    quick_signature,
    read_table,
    resolve_path,
    write_csv,
)


def inventory(config: dict) -> pd.DataFrame:
    rows: list[dict] = []
    roots = config.get("video_roots") or {}
    unexpected = set(roots) - ALLOWED_SOURCES
    if unexpected:
        raise ValueError(
            f"Unsupported video roots {sorted(unexpected)}. CAE-ROOM must not enter hand-motion recovery."
        )

    for source in ("gopro_left", "gopro_right", "cae_hand"):
        root = resolve_path(config, roots.get(source))
        if root is None:
            continue
        if not root.exists():
            rows.append(
                {
                    "source": source,
                    "video_path": str(root),
                    "inventory_status": "root_missing",
                    "error": "configured root does not exist",
                }
            )
            continue
        videos = sorted(
            (path for path in root.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS),
            key=natural_key,
        )
        for path in videos:
            base = {
                "source": source,
                "target_hand": "left" if source == "gopro_left" else "right" if source == "gopro_right" else "",
                "participant": infer_participant(path),
                "video_name": path.name,
                "video_path": str(path.resolve()),
                "file_size_bytes": path.stat().st_size,
                "modified_time_ns": path.stat().st_mtime_ns,
                "quick_signature": quick_signature(path),
                "inventory_status": "ok",
                "error": "",
            }
            try:
                base.update(ffprobe_metadata(path))
            except Exception as exc:  # inventory must continue and expose the failure
                base["inventory_status"] = "metadata_error"
                base["error"] = str(exc)
            rows.append(base)
    return pd.DataFrame(rows)


def compare_frozen(config: dict, current: pd.DataFrame) -> pd.DataFrame:
    frozen_path = resolve_path(config, config.get("frozen_video_inventory"))
    if frozen_path is None or not frozen_path.exists():
        return pd.DataFrame(
            [{"check": "frozen_inventory", "status": "not_compared", "detail": str(frozen_path or "") }]
        )

    frozen = read_table(frozen_path)
    lower = {normalize_name(column): column for column in frozen.columns}
    source_col = lower.get("source") or lower.get("camera") or lower.get("cameraside")
    path_col = lower.get("videopath") or lower.get("path") or lower.get("filepath")
    name_col = lower.get("videoname") or lower.get("filename") or lower.get("file")
    id_col = lower.get("videoid")
    if path_col:
        frozen["match_key"] = frozen[path_col].astype(str).map(lambda x: Path(x).name.lower())
    elif name_col:
        frozen["match_key"] = frozen[name_col].astype(str).map(lambda x: Path(x).name.lower())
    elif id_col:
        frozen["match_key"] = frozen[id_col].astype(str).str.lower()
    else:
        return pd.DataFrame(
            [{"check": "frozen_inventory", "status": "cannot_compare", "detail": "No video path/name/id column"}]
        )

    gopro = current[current["source"].isin(["gopro_left", "gopro_right"])].copy()
    gopro["match_key"] = gopro["video_name"].astype(str).str.lower()
    duplicate_keys = gopro[gopro.duplicated("match_key", keep=False)]["match_key"].unique()
    if len(duplicate_keys):
        if source_col:
            frozen["match_key"] = (
                frozen[source_col].astype(str).str.lower() + "|" + frozen["match_key"]
            )
            gopro["match_key"] = gopro["source"] + "|" + gopro["match_key"]
        else:
            return pd.DataFrame(
                [{"check": "frozen_inventory", "status": "cannot_compare", "detail": "Duplicate basenames and no source column"}]
            )

    frozen_keys = set(frozen["match_key"].dropna())
    current_keys = set(gopro["match_key"].dropna())
    rows = [
        {"check": "frozen_count", "status": "pass" if len(frozen) == 107 else "review", "detail": f"frozen rows={len(frozen)}; expected 107"},
        {"check": "current_gopro_count", "status": "pass" if len(gopro) == 107 else "review", "detail": f"current rows={len(gopro)}; expected 107"},
        {"check": "missing_from_current", "status": "pass" if frozen_keys <= current_keys else "fail", "detail": "; ".join(sorted(frozen_keys - current_keys))},
        {"check": "new_gopro_files", "status": "pass" if current_keys <= frozen_keys else "review", "detail": "; ".join(sorted(current_keys - frozen_keys))},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only inventory of GoPro and CAE-HAND videos.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    recovery_root = ensure_recovery_root(config)
    output_dir = recovery_root / "01_inventory"
    current = inventory(config)
    write_csv(current, output_dir / "video_inventory_recheck.csv")
    comparison = compare_frozen(config, current)
    write_csv(comparison, output_dir / "frozen_inventory_check.csv")

    summary = (
        current.groupby(["source", "inventory_status"], dropna=False)
        .agg(videos=("video_path", "count"), participants=("participant", lambda x: x[x != ""].nunique()))
        .reset_index()
    )
    write_csv(summary, output_dir / "inventory_summary.csv")
    print(f"Wrote read-only inventory outputs to {output_dir}")


if __name__ == "__main__":
    main()
