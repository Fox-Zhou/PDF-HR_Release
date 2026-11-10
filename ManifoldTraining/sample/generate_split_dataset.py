#!/usr/bin/env python3
"""Merge split-listed raw 97D motions without changing their features."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm


RANDOM_SEED = 42
FEATURE_DIM = 97
DEFAULT_CHUNK_FRAMES = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge split-listed raw (T, 97) motions into one (N, 97) memmap "
            "without changing quaternion yaw or any other feature."
        )
    )
    parser.add_argument(
        "--data-root", "--data_root", dest="data_root", type=Path, required=True,
        help="Root used to resolve relative paths in the split file.",
    )
    parser.add_argument(
        "--split-file", "--split_file", dest="split_file", type=Path, required=True,
        help="TXT file containing one NPY path relative to data-root per line.",
    )
    parser.add_argument(
        "--output-dir", "--output_dir", dest="output_dir", type=Path, required=True,
        help="Directory for the merged <split-name>.npy file.",
    )
    parser.add_argument(
        "--chunk-frames", type=int, default=DEFAULT_CHUNK_FRAMES,
        help=f"Frames copied per chunk (default: {DEFAULT_CHUNK_FRAMES:,}).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    parser.add_argument(
        "--no-shuffle", action="store_true",
        help="Preserve split-file order instead of deterministic shuffling.",
    )
    # Backward-compatible no-op: merging no longer performs device computation.
    parser.add_argument("--device", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def load_split_paths(split_file: Path, data_root: Path) -> list[Path]:
    paths: list[Path] = []
    with split_file.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            split_path = Path(value)
            if split_path.suffix.lower() != ".npy":
                raise ValueError(
                    f"Expected an .npy path at {split_file}:{line_number}, got {value!r}"
                )

            if split_path.is_absolute():
                raise ValueError(
                    f"Split paths must be relative to data-root at "
                    f"{split_file}:{line_number}, got {value!r}"
                )
            if ".." in split_path.parts:
                raise ValueError(
                    f"Split path must not escape data-root at "
                    f"{split_file}:{line_number}, got {value!r}"
                )

            path = (data_root / split_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Motion file not found in data-root: {value!r}")
            paths.append(path)
    if not paths:
        raise ValueError(f"Split contains no motion paths: {split_file}")
    return paths


def load_motion(path: Path) -> np.ndarray:
    motion = np.load(path, mmap_mode="r", allow_pickle=False)
    shape = motion.shape
    dtype = motion.dtype
    if motion.ndim != 2 or shape[1] != FEATURE_DIM:
        del motion
        raise ValueError(f"Expected motion shape (T, {FEATURE_DIM}), got {shape}: {path}")
    if shape[0] <= 0:
        del motion
        raise ValueError(f"Motion has no frames: {path}")
    if dtype != np.float32:
        del motion
        raise ValueError(f"Expected float32 motion, got {dtype}: {path}")
    return motion


def inspect_motions(paths: list[Path]) -> tuple[list[tuple[Path, int]], int]:
    clips: list[tuple[Path, int]] = []
    total_frames = 0
    for path in tqdm(paths, desc="Reading NPY headers"):
        motion = load_motion(path)
        num_frames = int(motion.shape[0])
        del motion
        clips.append((path, num_frames))
        total_frames += num_frames
    return clips, total_frames


def build_dataset(
    paths: list[Path], output_path: Path, overwrite: bool, chunk_frames: int
) -> int:
    if chunk_frames <= 0:
        raise ValueError(f"--chunk-frames must be positive, got {chunk_frames}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists; pass --overwrite to replace it: {output_path}"
        )

    clips, total_frames = inspect_motions(paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=(total_frames, FEATURE_DIM)
    )
    output_offset = 0
    try:
        for path, num_frames in tqdm(clips, desc="Merging raw motions"):
            motion = load_motion(path)
            for clip_start in range(0, num_frames, chunk_frames):
                clip_end = min(clip_start + chunk_frames, num_frames)
                output[output_offset + clip_start : output_offset + clip_end] = motion[
                    clip_start:clip_end
                ]
            output_offset += num_frames
            del motion
        output.flush()
    finally:
        del output
    return total_frames


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    split_file = args.split_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    paths = load_split_paths(split_file, data_root)
    if not args.no_shuffle:
        random.Random(RANDOM_SEED).shuffle(paths)
    output_path = output_dir / f"{split_file.stem}.npy"
    total_frames = build_dataset(paths, output_path, args.overwrite, args.chunk_frames)

    file_size_gib = output_path.stat().st_size / (1024**3)
    print(f"Saved: {output_path}")
    print(f"Motions: {len(paths)}")
    print(f"Shape: ({total_frames}, {FEATURE_DIM})")
    print(f"Size: {file_size_gib:.3f} GiB")
    print("Feature conversion: none (absolute yaw preserved)")


if __name__ == "__main__":
    main()
