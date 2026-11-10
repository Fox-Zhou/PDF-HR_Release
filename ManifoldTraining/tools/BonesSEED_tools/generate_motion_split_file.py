#!/usr/bin/env python3
"""Generate reproducible split files for preprocessed BONES-SEED G1 NPY clips."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm


RANDOM_SEED = 42
EXPECTED_FEATURE_DIM = 97
DEFAULT_SPLIT_NAME = "bonesseed_split_100percent"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MotionMetadata:
    package: str
    is_mirror: bool


@dataclass(frozen=True)
class MotionSequence:
    path: Path
    rel_path: Path
    num_frames: int
    package: str
    is_mirror: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate split TXT files for preprocessed BONES-SEED (T, 97) NPY motions."
    )
    parser.add_argument(
        "--data-root",
        "--data_root",
        dest="data_root",
        type=Path,
        required=True,
        help="Root containing preprocessed NPY motions, usually BonesSEED__G1_processed.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=(
            "BONES-SEED metadata CSV. By default, infer "
            "<data-root-parent>/metadata/seed_metadata_v004.csv."
        ),
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
        default=DEFAULT_SPLIT_NAME,
        help="Output TXT basename without extension.",
    )
    parser.add_argument(
        "--target-samples",
        "--target_samples",
        dest="target_samples",
        default="30_000_000",
        help="Target frame count, or 'all' for every eligible motion.",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Distribute the target frame count across packages as evenly as possible.",
    )
    parser.add_argument(
        "--include-package",
        nargs="*",
        default=[],
        help="Only include these packages. Empty means all packages.",
    )
    parser.add_argument(
        "--exclude-keyword",
        nargs="*",
        default=[],
        help="Exclude motions whose filename contains any listed substring.",
    )
    parser.add_argument(
        "--include-mirror",
        action="store_true",
        help="Include mirrored motions. They are excluded by default.",
    )
    parser.add_argument(
        "--no-shuffle-output",
        action="store_true",
        help="Keep selected output paths in selection order instead of shuffling them.",
    )
    return parser.parse_args()


def parse_target_samples(value: str) -> int | None:
    normalized = str(value).replace("_", "").strip().lower()
    if normalized == "all":
        return None
    try:
        target = int(normalized)
    except ValueError as exc:
        raise ValueError(f"--target-samples must be a positive integer or 'all', got {value!r}") from exc
    if target <= 0:
        raise ValueError(f"--target-samples must be positive, got {target}")
    return target


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def infer_metadata_path(data_root: Path) -> Path:
    return data_root.parent / "metadata" / "seed_metadata_v004.csv"


def load_metadata(metadata_path: Path) -> dict[str, MotionMetadata]:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    result: dict[str, MotionMetadata] = {}
    with metadata_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"filename", "package", "is_mirror"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Metadata CSV missing columns: {sorted(missing)}")
        for row in reader:
            filename = str(row["filename"]).strip()
            metadata = MotionMetadata(
                package=str(row["package"]).strip(),
                is_mirror=parse_bool(row["is_mirror"]),
            )
            previous = result.get(filename)
            if previous is not None and previous != metadata:
                raise ValueError(
                    f"Conflicting metadata for filename {filename!r}: {previous} vs {metadata}"
                )
            result[filename] = metadata
    return result


def read_motion_shape(path: Path) -> tuple[int, int]:
    motion = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        if motion.ndim != 2:
            raise ValueError(f"Expected a 2D motion matrix, got shape {motion.shape}: {path}")
        shape = (int(motion.shape[0]), int(motion.shape[1]))
    finally:
        del motion
    if shape[0] <= 0:
        raise ValueError(f"Motion has no frames: {path}")
    if shape[1] != EXPECTED_FEATURE_DIM:
        raise ValueError(
            f"Expected feature dimension {EXPECTED_FEATURE_DIM}, got {shape[1]}: {path}"
        )
    return shape


def discover_motion_sequences(
    data_root: Path,
    metadata_by_filename: dict[str, MotionMetadata],
    include_packages: set[str],
    exclude_keywords: list[str],
    include_mirror: bool,
) -> list[MotionSequence]:
    motion_paths = sorted(data_root.rglob("*.npy"))
    if not motion_paths:
        raise FileNotFoundError(f"No NPY motion files found under {data_root}")

    sequences: list[MotionSequence] = []
    skipped = defaultdict(int)
    for path in tqdm(motion_paths, desc="Reading NPY headers"):
        if any(keyword in path.stem for keyword in exclude_keywords):
            skipped["excluded keyword"] += 1
            continue

        metadata = metadata_by_filename.get(path.stem)
        if metadata is None:
            skipped["missing metadata"] += 1
            continue
        if include_packages and metadata.package not in include_packages:
            skipped["package filter"] += 1
            continue
        if metadata.is_mirror and not include_mirror:
            skipped["mirrored"] += 1
            continue

        num_frames, _ = read_motion_shape(path)
        sequences.append(
            MotionSequence(
                path=path,
                rel_path=path.relative_to(data_root),
                num_frames=num_frames,
                package=metadata.package,
                is_mirror=metadata.is_mirror,
            )
        )

    for reason, count in sorted(skipped.items()):
        print(f"Skipped ({reason}): {count}")
    if not sequences:
        raise ValueError("No eligible motion sequences remain after filtering")
    return sequences


def choose_random_split(
    sequences: list[MotionSequence], target_frames: int
) -> list[MotionSequence]:
    shuffled = list(sequences)
    random.Random(RANDOM_SEED).shuffle(shuffled)
    selected: list[MotionSequence] = []
    current_frames = 0
    for sequence in shuffled:
        current_error = abs(target_frames - current_frames)
        next_error = abs(target_frames - (current_frames + sequence.num_frames))
        if next_error <= current_error:
            selected.append(sequence)
            current_frames += sequence.num_frames
        else:
            break
    return selected


def allocate_balanced_quotas(
    group_totals: dict[str, int], target_frames: int
) -> dict[str, int]:
    groups = sorted(group_totals, key=lambda group: group_totals[group])
    remaining_target = min(target_frames, sum(group_totals.values()))
    quotas: dict[str, int] = {}
    for index, group in enumerate(groups):
        remaining_groups = len(groups) - index
        quota = remaining_target // remaining_groups
        quotas[group] = min(group_totals[group], quota)
        remaining_target -= quotas[group]
    return quotas


def choose_balanced_split(
    sequences: list[MotionSequence], target_frames: int
) -> list[MotionSequence]:
    by_package: dict[str, list[MotionSequence]] = defaultdict(list)
    for sequence in sequences:
        by_package[sequence.package].append(sequence)

    totals = {
        package: sum(sequence.num_frames for sequence in package_sequences)
        for package, package_sequences in by_package.items()
    }
    quotas = allocate_balanced_quotas(totals, target_frames)
    print(f"Available frames by package: {totals}")
    print(f"Target quotas by package: {quotas}")

    rng = random.Random(RANDOM_SEED)
    selected: list[MotionSequence] = []
    for package in sorted(by_package):
        candidates = list(by_package[package])
        rng.shuffle(candidates)
        package_selected = choose_closest_prefix(candidates, quotas[package])
        selected.extend(package_selected)
        print(
            f"[{package}] selected "
            f"{sum(sequence.num_frames for sequence in package_selected)} frames"
        )
    return selected


def choose_closest_prefix(
    candidates: list[MotionSequence], target_frames: int
) -> list[MotionSequence]:
    selected: list[MotionSequence] = []
    current_frames = 0
    for sequence in candidates:
        next_frames = current_frames + sequence.num_frames
        if current_frames < target_frames or abs(target_frames - next_frames) <= abs(
            target_frames - current_frames
        ):
            selected.append(sequence)
            current_frames = next_frames
            if current_frames >= target_frames:
                break
        else:
            break
    return selected


def save_split(
    selected: list[MotionSequence], output_dir: Path, split_name: str, shuffle: bool
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split_name}.txt"
    output_sequences = list(selected)
    if shuffle:
        random.Random(RANDOM_SEED).shuffle(output_sequences)
    with output_path.open("w", encoding="utf-8") as stream:
        for sequence in output_sequences:
            stream.write(f"{sequence.rel_path.as_posix()}\n")
    return output_path


def print_summary(all_sequences: list[MotionSequence], selected: list[MotionSequence]) -> None:
    total_frames = sum(sequence.num_frames for sequence in all_sequences)
    selected_frames = sum(sequence.num_frames for sequence in selected)
    by_package: dict[str, list[int]] = defaultdict(list)
    for sequence in selected:
        by_package[sequence.package].append(sequence.num_frames)

    print(f"Eligible sequences: {len(all_sequences)}")
    print(f"Eligible frames: {total_frames}")
    print(f"Selected sequences: {len(selected)}")
    print(f"Selected frames: {selected_frames}")
    print(f"Selected ratio: {selected_frames / total_frames * 100:.3f}%")
    print("Selected package distribution:")
    for package in sorted(by_package):
        frames = sum(by_package[package])
        print(f"  {package}: sequences={len(by_package[package])}, frames={frames}")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    metadata_path = (
        args.metadata.expanduser().resolve()
        if args.metadata is not None
        else infer_metadata_path(data_root)
    )
    target_frames = parse_target_samples(args.target_samples)
    metadata_by_filename = load_metadata(metadata_path)
    sequences = discover_motion_sequences(
        data_root=data_root,
        metadata_by_filename=metadata_by_filename,
        include_packages=set(args.include_package),
        exclude_keywords=args.exclude_keyword,
        include_mirror=args.include_mirror,
    )

    if target_frames is None or target_frames >= sum(s.num_frames for s in sequences):
        selected = list(sequences)
    elif args.balance:
        selected = choose_balanced_split(sequences, target_frames)
    else:
        selected = choose_random_split(sequences, target_frames)

    output_path = save_split(
        selected,
        args.output_dir.expanduser().resolve(),
        args.split_name,
        shuffle=not args.no_shuffle_output,
    )
    print(f"Saved: {output_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Include packages: {args.include_package or 'all'}")
    print(f"Include mirror: {args.include_mirror}")
    print_summary(sequences, selected)


if __name__ == "__main__":
    main()
