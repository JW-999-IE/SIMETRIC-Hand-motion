from __future__ import annotations

import argparse
import io
import math
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from common import ensure_recovery_root, load_config, read_table, safe_float


def frame_at(video_path: Path, second: float, width: int = 480) -> Image.Image:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found. Install FFmpeg and put it on PATH.")
    command = [
        ffmpeg,
        "-v",
        "error",
        "-ss",
        f"{max(second, 0):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        f"scale={width}:-2",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    completed = subprocess.run(command, check=True, capture_output=True)
    return Image.open(io.BytesIO(completed.stdout)).convert("RGB")


def sample_times(row: pd.Series, count: int, prep_skip_seconds: float) -> list[float]:
    duration = safe_float(row.get("video_duration_sec"), 0.0)
    start = safe_float(row.get("video_local_start_sec"), float("nan"))
    end = safe_float(row.get("video_local_end_sec"), float("nan"))
    if math.isnan(start) or math.isnan(end) or end <= start:
        start = min(max(prep_skip_seconds, 0.0), max(duration - 1.0, 0.0))
        end = max(duration - 0.5, start + 0.5)
    if count <= 1:
        return [(start + end) / 2]
    return [start + index * (end - start) / (count - 1) for index in range(count)]


def make_sheet(row: pd.Series, output: Path, count: int, prep_skip_seconds: float) -> None:
    video = Path(str(row["video_path"]))
    times = sample_times(row, count, prep_skip_seconds)
    tiles: list[Image.Image] = []
    font = ImageFont.load_default()
    for second in times:
        image = frame_at(video, second)
        canvas = Image.new("RGB", (image.width, image.height + 28), "white")
        canvas.paste(image, (0, 28))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 7), f"video-local time {second:.2f} s", fill="black", font=font)
        tiles.append(canvas)

    columns = 3
    rows = math.ceil(len(tiles) / columns)
    header_height = 66
    width = columns * tiles[0].width
    height = header_height + rows * tiles[0].height
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    title = (
        f"{row.get('review_id', '')} | {row.get('attempt_id', '')} | "
        f"{row.get('participant', '')} | {row.get('source', '')} | target {row.get('target_hand', '')}"
    )
    draw.text((8, 8), title, fill="black", font=font)
    draw.text((8, 28), video.name, fill="black", font=font)
    draw.text((8, 47), "Use visible procedural evidence; workbook timestamps are reference-only.", fill="black", font=font)
    for index, tile in enumerate(tiles):
        x = (index % columns) * tile.width
        y = header_height + (index // columns) * tile.height
        sheet.paste(tile, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visual contact sheets for mapping review.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--queue", default="")
    parser.add_argument("--review-id", default="", help="Optional single review_id")
    parser.add_argument("--participant", default="", help="Optional participant filter, for example P4")
    parser.add_argument("--target-hand", choices=["left", "right"], default="")
    parser.add_argument("--source", choices=["gopro_left", "gopro_right", "cae_hand"], default="")
    parser.add_argument("--max-rows", type=int, default=50)
    parser.add_argument("--frames", type=int, default=9)
    args = parser.parse_args()

    config = load_config(args.config)
    recovery_root = ensure_recovery_root(config)
    queue_path = (
        Path(args.queue).expanduser().resolve()
        if args.queue
        else recovery_root / "02_mapping_review" / "mapping_review_queue.csv"
    )
    queue = read_table(queue_path)
    queue = queue[queue["video_path"].fillna("").astype(str).ne("")].copy()
    if args.participant:
        queue = queue[queue["participant"].astype(str).str.upper().eq(args.participant.upper())]
    if args.target_hand:
        queue = queue[queue["target_hand"].astype(str).str.lower().eq(args.target_hand)]
    if args.source:
        queue = queue[queue["source"].astype(str).str.lower().eq(args.source)]
    if args.review_id:
        queue = queue[queue["review_id"].astype(str).eq(args.review_id)]
    elif args.participant or args.target_hand or args.source:
        queue = queue.head(args.max_rows)
    else:
        queue = queue[
            queue["mapping_status"].fillna("").astype(str).str.lower().isin(
                ["needs_video_review", "existing_mapping_review", "ambiguous", "unresolved"]
            )
        ].head(args.max_rows)

    prep_skip_seconds = float((config.get("review") or {}).get("default_prep_skip_seconds", 0))
    output_dir = recovery_root / "02_mapping_review" / "contact_sheets"
    failures: list[dict] = []
    for _, row in queue.iterrows():
        try:
            make_sheet(row, output_dir / f"{row['review_id']}.jpg", args.frames, prep_skip_seconds)
        except Exception as exc:
            failures.append({"review_id": row.get("review_id", ""), "error": str(exc)})
    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "contact_sheet_failures.csv", index=False)
    print(f"Generated {len(queue) - len(failures)} contact sheets in {output_dir}")


if __name__ == "__main__":
    main()
