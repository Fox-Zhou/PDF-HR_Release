#!/usr/bin/env python3
"""Generate a filename-filtered motion split for preprocessed LaFAN1 G1 clips."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


# A motion is included when its filename contains any of these fields.
# Matching is case-insensitive. Leave the list empty to include every motion.
INCLUDE_FIELDS: list[str] = ['walk', 'run', 'sprint',]  # left empty to include every motion

EXPECTED_FEATURE_DIM = 97
DEFAULT_SPLIT_NAME = "lafan"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a TXT split from preprocessed LaFAN1 G1 (T, 97) NPY "
            "motions using the hardcoded INCLUDE_FIELDS filename filter."
        )
    )
    parser.add_argument(
        "--data-root",
        "--data_root",
        dest="data_root",
        type=Path,
        required=True,
        help="Root directory containing preprocessed LaFAN1 NPY motions.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        default=PROJECT_ROOT / "sample" / "split_files",
        help="Directory in which to write the split TXT file.",
    )
    parser.add_argument(
        "--split-name",
        "--split_name",
        dest="split_name",
        default=DEFAULT_SPLIT_NAME,
        help="Output TXT basename without extension.",
    )
    return parser.parse_args()


def normalized_include_fields() -> tuple[str, ...]:
    fields = tuple(field.strip().casefold() for field in INCLUDE_FIELDS if field.strip())
    if len(fields) != len(INCLUDE_FIELDS):
        raise ValueError("INCLUDE_FIELDS must not contain empty or whitespace-only fields")
    return fields


def filename_is_included(path: Path, fields: tuple[str, ...]) -> bool:
    return not fields or any(field in path.stem.casefold() for field in fields)


def read_motion_frames(path: Path) -> int:
    motion = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        if motion.ndim != 2 or motion.shape[1] != EXPECTED_FEATURE_DIM:
            raise ValueError(
                f"Expected shape (T, {EXPECTED_FEATURE_DIM}), got {motion.shape}: {path}"
            )
        frames = int(motion.shape[0])
        dtype = motion.dtype
    finally:
        del motion
    if frames <= 0:
        raise ValueError(f"Motion has no frames: {path}")
    if dtype != np.float32:
        raise ValueError(f"Expected float32 motion, got {dtype}: {path}")
    return frames


def discover_selected_motions(
    data_root: Path, fields: tuple[str, ...]
) -> list[tuple[Path, int]]:
    motion_paths = sorted(data_root.rglob("*.npy"))
    if not motion_paths:
        raise FileNotFoundError(f"No NPY motion files found under {data_root}")

    selected: list[tuple[Path, int]] = []
    for path in tqdm(motion_paths, desc="Reading LaFAN1 NPY headers"):
        if not filename_is_included(path, fields):
            continue
        frames = read_motion_frames(path)
        selected.append((path.relative_to(data_root), frames))
    if not selected:
        raise ValueError(
            f"No motion filename matched INCLUDE_FIELDS={list(INCLUDE_FIELDS)!r}"
        )
    return selected


def save_split(
    selected: list[tuple[Path, int]], output_dir: Path, split_name: str
) -> Path:
    if not split_name or Path(split_name).name != split_name:
        raise ValueError(f"split-name must be a filename stem, got {split_name!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split_name}.txt"
    with output_path.open("w", encoding="utf-8") as stream:
        for relative_path, _ in selected:
            stream.write(f"{relative_path.as_posix()}\n")
    return output_path


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    fields = normalized_include_fields()
    selected = discover_selected_motions(data_root, fields)
    output_path = save_split(
        selected,
        args.output_dir.expanduser().resolve(),
        args.split_name,
    )
    total_frames = sum(frames for _, frames in selected)
    print(f"INCLUDE_FIELDS: {list(INCLUDE_FIELDS) if fields else 'all'}")
    print(f"Selected motions: {len(selected)}")
    print(f"Selected frames: {total_frames}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
