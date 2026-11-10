#!/usr/bin/env python3
"""Plot the distance distribution of a pose, velocity, or acceleration sample NPY."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


SAMPLE_SCHEMAS = {
    34: ("pose", 33, 32),
    66: ("velocity", 65, 64),
    98: ("acceleration", 97, 96),
}
DEFAULT_BINS = 100
DEFAULT_CHUNK_ROWS = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw a histogram of either distance column in a generated pose, "
            "velocity, or acceleration sample NPY."
        )
    )
    parser.add_argument(
        "sample_file",
        type=Path,
        help="Sample NPY with shape (B, 34), (B, 66), or (B, 98).",
    )
    parser.add_argument(
        "--distance-kind",
        choices=("29dof", "root+29dof"),
        default="29dof",
        help="Distance to plot (default: 29dof).",
    )
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument(
        "--min-distance",
        type=float,
        default=None,
        help="Lower histogram bound. Defaults to the data minimum.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="Upper histogram bound. Defaults to the data maximum.",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=DEFAULT_CHUNK_ROWS,
        help="Rows processed at once (default: 1,000,000).",
    )
    parser.add_argument(
        "--density",
        action="store_true",
        help="Plot probability density instead of sample counts.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive plot window after saving the image.",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def validate_sample(sample: np.ndarray, source: Path) -> tuple[str, int, int]:
    if sample.ndim != 2 or sample.shape[1] not in SAMPLE_SCHEMAS:
        supported = ", ".join(str(width) for width in SAMPLE_SCHEMAS)
        raise ValueError(
            f"Unsupported sample shape {sample.shape} in {source}; "
            f"expected a 2D array with {supported} columns"
        )
    if sample.shape[0] <= 0:
        raise ValueError(f"Sample file has no rows: {source}")
    if not np.issubdtype(sample.dtype, np.number):
        raise ValueError(f"Sample array must be numeric, got {sample.dtype}")
    return SAMPLE_SCHEMAS[int(sample.shape[1])]


def scan_distance_range(
    sample: np.ndarray, distance_index: int, chunk_rows: int
) -> tuple[float, float, int, float]:
    minimum = np.inf
    maximum = -np.inf
    total = 0
    value_sum = 0.0
    for start in range(0, len(sample), chunk_rows):
        values = np.asarray(
            sample[start : start + chunk_rows, distance_index], dtype=np.float64
        )
        if not np.isfinite(values).all():
            bad_count = int(values.size - np.count_nonzero(np.isfinite(values)))
            raise ValueError(
                f"Distance column contains {bad_count} NaN/inf values in rows "
                f"[{start}:{start + len(values)}]"
            )
        minimum = min(minimum, float(values.min()))
        maximum = max(maximum, float(values.max()))
        total += int(values.size)
        value_sum += float(values.sum(dtype=np.float64))
    return minimum, maximum, total, value_sum / total


def histogram_chunked(
    sample: np.ndarray,
    distance_index: int,
    edges: np.ndarray,
    chunk_rows: int,
) -> tuple[np.ndarray, int, int]:
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    below = 0
    above = 0
    lower, upper = float(edges[0]), float(edges[-1])
    for start in range(0, len(sample), chunk_rows):
        values = np.asarray(
            sample[start : start + chunk_rows, distance_index], dtype=np.float64
        )
        below += int(np.count_nonzero(values < lower))
        above += int(np.count_nonzero(values > upper))
        counts += np.histogram(values, bins=edges)[0]
    return counts, below, above


def main() -> None:
    args = parse_args()
    if args.bins <= 0 or args.chunk_rows <= 0 or args.dpi <= 0:
        raise ValueError("bins, chunk-rows, and dpi must be positive")

    sample_path = args.sample_file.expanduser().resolve()
    if not sample_path.is_file():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")
    sample = np.load(sample_path, mmap_mode="r", allow_pickle=False)
    sample_type, joint_distance_index, combined_distance_index = validate_sample(
        sample, sample_path
    )
    distance_index = (
        joint_distance_index
        if args.distance_kind == "29dof"
        else combined_distance_index
    )
    data_min, data_max, total, mean = scan_distance_range(
        sample, distance_index, args.chunk_rows
    )

    plot_min = data_min if args.min_distance is None else float(args.min_distance)
    plot_max = data_max if args.max_distance is None else float(args.max_distance)
    if not np.isfinite(plot_min) or not np.isfinite(plot_max):
        raise ValueError("Histogram bounds must be finite")
    if plot_min > plot_max:
        raise ValueError(
            f"min-distance ({plot_min}) must not exceed max-distance ({plot_max})"
        )
    if plot_min == plot_max:
        padding = max(abs(plot_min) * 0.01, 0.5)
        plot_min -= padding
        plot_max += padding

    edges = np.linspace(plot_min, plot_max, args.bins + 1, dtype=np.float64)
    counts, below, above = histogram_chunked(
        sample, distance_index, edges, args.chunk_rows
    )
    plotted = int(counts.sum())
    heights = counts.astype(np.float64)
    y_label = "Sample count"
    if args.density:
        widths = np.diff(edges)
        if plotted > 0:
            heights /= plotted * widths
        y_label = "Probability density"

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(__file__).resolve().parent / "distribution.png"

    summary = (
        f"N={total:,}   mean={mean:.6g}   min={data_min:.6g}   max={data_max:.6g}"
    )
    if below or above:
        summary += f"\noutside plot: below={below:,}, above={above:,}"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        edges[:-1],
        heights,
        width=np.diff(edges),
        align="edge",
        color="#4472C4",
        edgecolor="white",
        linewidth=0.35,
    )
    ax.set_xlabel(f"{args.distance_kind} distance")
    ax.set_ylabel(y_label)
    ax.set_title(
        f"{sample_type.capitalize()} {args.distance_kind} distance distribution\n"
        f"{summary}",
        pad=12,
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(
        f"type={sample_type}, shape={sample.shape}, distance={args.distance_kind}, "
        f"distance_column={distance_index}, "
        f"range=[{data_min:.9g}, {data_max:.9g}], mean={mean:.9g}, "
        f"plotted={plotted}/{total}"
    )
    print(f"Saved histogram: {output_path}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
