from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import re
from typing import Iterable

import numpy as np
import pandas as pd


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}


def natural_key(value: object):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", str(value))
    ]


def participant_number(value: object) -> int:
    m = re.search(r"(\d+)", str(value))
    return int(m.group(1)) if m else 999999


def stable_video_id(path: Path) -> str:
    # Stable across runs as long as the path does not change.
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_participants(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    out = set()
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                out.add(item.upper())
    return out or None


def discover_cae_hand_videos(root: Path, participants: set[str] | None = None) -> list[Path]:
    paths: list[Path] = []
    if not root.exists():
        return paths
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if "-HAND" not in path.stem.upper():
            continue
        participant = infer_participant(path)
        if participants and participant.upper() not in participants:
            continue
        paths.append(path)
    return sorted(paths, key=lambda p: natural_key(str(p)))


def discover_cae_room_videos(root: Path, participants: set[str] | None = None) -> list[Path]:
    paths: list[Path] = []
    if not root.exists():
        return paths
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if "-ROOM" not in path.stem.upper():
            continue
        participant = infer_participant(path)
        if participants and participant.upper() not in participants:
            continue
        paths.append(path)
    return sorted(paths, key=lambda p: natural_key(str(p)))


def infer_participant(path: Path) -> str:
    # Prefer folder name Pxx.
    for part in reversed(path.parts):
        if re.fullmatch(r"P\d+", part, flags=re.I):
            return part.upper()
    m = re.search(r"\bP(\d+)\b", path.name, flags=re.I)
    if m:
        return f"P{int(m.group(1))}"
    m = re.search(r"CAE-P(\d+)", path.name, flags=re.I)
    if m:
        return f"P{int(m.group(1))}"
    return "UNKNOWN"


def write_table(df: pd.DataFrame, path: Path) -> Path:
    ensure_dir(path.parent)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception:
            fallback = path.with_suffix(".csv.gz")
            df.to_csv(fallback, index=False, compression="gzip")
            return fallback
    df.to_csv(path, index=False)
    return path


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".gz":
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


def read_part_files(folder: Path) -> pd.DataFrame:
    parquet = sorted(folder.glob("part-*.parquet"))
    csvgz = sorted(folder.glob("part-*.csv.gz"))
    csv = sorted(folder.glob("part-*.csv"))
    paths = parquet or csvgz or csv
    if not paths:
        return pd.DataFrame()
    frames = [read_table(path) for path in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    # angle ABC
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba == 0 or nbc == 0:
        return float("nan")
    cosv = float(np.dot(ba, bc) / (nba * nbc))
    cosv = float(np.clip(cosv, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosv)))


def euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def robust_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")
