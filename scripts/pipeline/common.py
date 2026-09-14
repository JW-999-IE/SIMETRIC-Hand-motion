from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mts", ".m2ts"}
ALLOWED_SOURCES = {"gopro_left", "gopro_right", "cae_hand"}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(config_path)
    return config


def resolve_path(config: dict[str, Any], value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path(config["_config_path"]).parent / path
    return path.resolve()


def ensure_recovery_root(config: dict[str, Any]) -> Path:
    recovery_root = resolve_path(config, config.get("recovery_root"))
    frozen_root = resolve_path(config, config.get("frozen_gopro_output_root"))
    if recovery_root is None:
        raise ValueError("config.recovery_root is required")
    if frozen_root is not None:
        try:
            recovery_root.relative_to(frozen_root)
        except ValueError:
            pass
        else:
            raise ValueError(
                "recovery_root must be physically separate from frozen_gopro_output_root"
            )
        if recovery_root == frozen_root:
            raise ValueError("recovery_root cannot equal frozen_gopro_output_root")
    recovery_root.mkdir(parents=True, exist_ok=True)
    return recovery_root


def read_table(path: str | Path, sheet_name: str | int | None = 0) -> pd.DataFrame:
    table_path = Path(path).expanduser().resolve()
    suffix = table_path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(table_path, sheet_name=sheet_name)
    if suffix == ".csv":
        return pd.read_csv(table_path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(table_path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(table_path)
    raise ValueError(f"Unsupported table type: {table_path}")


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    return output


def canonicalize_columns(
    df: pd.DataFrame,
    explicit_mapping: dict[str, str] | None,
    aliases: dict[str, Iterable[str]],
    required: Iterable[str] = (),
) -> pd.DataFrame:
    result = df.copy()
    explicit_mapping = explicit_mapping or {}
    normalized = {normalize_name(column): column for column in result.columns}
    rename: dict[str, str] = {}
    for canonical, source in explicit_mapping.items():
        if source in result.columns:
            rename[source] = canonical
        elif normalize_name(source) in normalized:
            rename[normalized[normalize_name(source)]] = canonical
    result = result.rename(columns=rename)

    normalized = {normalize_name(column): column for column in result.columns}
    for canonical, candidates in aliases.items():
        if canonical in result.columns:
            continue
        for candidate in candidates:
            source = normalized.get(normalize_name(candidate))
            if source is not None:
                result = result.rename(columns={source: canonical})
                break

    missing = [column for column in required if column not in result.columns]
    if missing:
        raise ValueError(
            f"Missing required canonical columns {missing}. Available: {list(df.columns)}"
        )
    return result


def normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def normalize_participant(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    match = re.search(r"(?i)(?:participant\s*)?p?\s*0*(\d{1,3})", text)
    return f"P{int(match.group(1))}" if match else text.upper()


def infer_participant(path: str | Path) -> str:
    text = str(path)
    matches = re.findall(r"(?i)(?:^|[\\/_\s-])P\s*0*(\d{1,3})(?=$|[\\/_\s-])", text)
    if matches:
        return f"P{int(matches[-1])}"
    matches = re.findall(r"(?i)(?:participant|part)[_\s-]*0*(\d{1,3})", text)
    return f"P{int(matches[-1])}" if matches else ""


def normalize_source(value: Any) -> str:
    text = normalize_name(value)
    if "gopro" in text and "left" in text:
        return "gopro_left"
    if "gopro" in text and "right" in text:
        return "gopro_right"
    if "cae" in text and ("hand" in text or "close" in text):
        return "cae_hand"
    if text in {"left", "l"}:
        return "gopro_left"
    if text in {"right", "r"}:
        return "gopro_right"
    return str(value).strip().lower()


def source_target_hand(source: str) -> str:
    source = normalize_source(source)
    if source == "gopro_left":
        return "left"
    if source == "gopro_right":
        return "right"
    return ""


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None or pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def ffprobe_metadata(path: str | Path) -> dict[str, Any]:
    executable = shutil.which("ffprobe")
    if executable is None:
        raise RuntimeError("ffprobe was not found. Install FFmpeg and put it on PATH.")
    command = [
        executable,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames:format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    stream = (payload.get("streams") or [{}])[0]
    fmt = payload.get("format") or {}

    def fraction(value: Any) -> float:
        text = str(value or "0/1")
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            return float(numerator) / float(denominator or 1)
        return float(text)

    return {
        "duration_sec": safe_float(fmt.get("duration")),
        "fps": fraction(stream.get("avg_frame_rate") or stream.get("r_frame_rate")),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "frame_count": int(stream.get("nb_frames") or 0),
    }


def quick_signature(path: str | Path, chunk_bytes: int = 1024 * 1024) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    size = file_path.stat().st_size
    digest.update(str(size).encode("ascii"))
    with file_path.open("rb") as handle:
        digest.update(handle.read(chunk_bytes))
        if size > chunk_bytes:
            handle.seek(max(size - chunk_bytes, 0))
            digest.update(handle.read(chunk_bytes))
    return digest.hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def natural_key(value: Any) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def assert_unique(df: pd.DataFrame, columns: list[str], label: str) -> None:
    duplicates = df[df.duplicated(columns, keep=False)]
    if not duplicates.empty:
        sample = duplicates[columns].head(10).to_dict("records")
        raise ValueError(f"{label} contains duplicate keys {columns}: {sample}")


ATTEMPT_ALIASES = {
    "attempt_id": ["Attempt_ID", "attempt id", "record_id", "procedure_id"],
    "participant": ["Participant", "participant_id", "subject", "clinician"],
    "configuration": ["configuration", "device", "technique", "LPC", "group"],
    "aborted": ["aborted", "abort", "is_aborted"],
}


MAPPING_ALIASES = {
    "attempt_id": ATTEMPT_ALIASES["attempt_id"],
    "participant": ATTEMPT_ALIASES["participant"],
    "configuration": ATTEMPT_ALIASES["configuration"],
    "source": ["source", "motion_source", "camera", "camera_side"],
    "target_hand": ["target_hand", "anatomical_hand", "hand"],
    "video_path": ["video_path", "path", "file", "file_path", "candidate_video"],
    "mapping_status": ["mapping_status", "map_status", "status", "decision"],
    "coverage_state": ["coverage_state", "video_coverage", "visibility_state"],
    "video_local_start_sec": ["video_local_start_sec", "start_sec", "video_start_sec"],
    "video_local_end_sec": ["video_local_end_sec", "end_sec", "video_end_sec"],
    "mapping_confidence": ["mapping_confidence", "confidence"],
    "review_notes": ["review_notes", "notes", "comment"],
}


QC_ALIASES = {
    "attempt_id": ATTEMPT_ALIASES["attempt_id"],
    "participant": ATTEMPT_ALIASES["participant"],
    "configuration": ATTEMPT_ALIASES["configuration"],
    "source": MAPPING_ALIASES["source"],
    "target_hand": MAPPING_ALIASES["target_hand"],
    "target_detection_coverage": [
        "target_detection_coverage",
        "target_hand_detection_coverage",
        "target_coverage",
        "primary_hand_coverage",
    ],
    "target_motion_coverage": [
        "target_motion_coverage",
        "motion_coverage",
        "wrist_motion_coverage",
    ],
    "temporal_bin_coverage": [
        "temporal_bin_coverage",
        "time_bin_coverage",
        "trajectory_bin_coverage",
    ],
    "longest_gap_fraction": [
        "longest_gap_fraction",
        "max_gap_fraction",
        "longest_observation_gap_fraction",
    ],
    "finite_wrist_points": ["finite_wrist_points", "n_wrist_points", "usable_wrist_points"],
    "identity_consistency": ["identity_consistency", "hand_identity_consistency"],
    "ambiguous_fraction": ["ambiguous_fraction", "ambiguous_assignment_fraction"],
    "side_switch_count": ["side_switch_count", "assignment_side_switch_count"],
    "hard_qc": ["hard_qc", "hard_qc_status", "hard_qc_fail", "identity_qc_fail"],
    "trajectory_qc_pass": ["trajectory_qc_pass", "trajectory_pass", "temporal_qc_pass"],
    "landmark_path": ["landmark_path", "trajectory_path", "landmarks_file"],
}
