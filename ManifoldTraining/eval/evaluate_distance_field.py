#!/usr/bin/env python3
"""Stream a motion dataset through one PDF-HR distance-field checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ManifoldTraining.sample.root_features import convert_root_features
from PDFHR.model import load_distance_field, torch_load_safe


DEFAULT_BINS = (0.0, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
FIELD_INPUT_DIMS = {"pose": 32, "vel": 64, "acc": 96}


@dataclass
class RunningMoments:
    count: int = 0
    total: float = 0.0
    total_square: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, values: np.ndarray) -> None:
        values64 = np.asarray(values, dtype=np.float64)
        self.count += int(values64.size)
        self.total += float(values64.sum(dtype=np.float64))
        self.total_square += float(np.square(values64).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(values64.min()))
        self.maximum = max(self.maximum, float(values64.max()))

    def export(self) -> dict[str, float | int]:
        if self.count == 0:
            raise ValueError("Cannot export empty statistics")
        mean = self.total / self.count
        variance = max(0.0, self.total_square / self.count - mean * mean)
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": mean,
            "std": math.sqrt(variance),
        }


def parse_bins(text: str) -> np.ndarray:
    try:
        values = [float(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("--bins must be comma-separated numbers") from error
    if len(values) < 2 or not np.isfinite(values).all():
        raise argparse.ArgumentTypeError("--bins needs at least two finite boundaries")
    edges = np.asarray(values, dtype=np.float64)
    if np.any(edges[1:] <= edges[:-1]):
        raise argparse.ArgumentTypeError("--bins boundaries must be strictly increasing")
    return edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a pose/vel/acc PDF-HR checkpoint on one NPY dataset or a directory "
            "of NPY shards, then write JSON and print an interval-count table."
        )
    )
    parser.add_argument("--checkpoint", "--ckpt", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: eval/results/<dataset>_<field>.json)",
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument(
        "--input-layout",
        choices=("auto", "raw", "field"),
        default="auto",
        help=(
            "auto treats 97 columns as raw merged dataset order; field treats rows "
            "as reordered model input"
        ),
    )
    parser.add_argument(
        "--bins",
        type=parse_bins,
        default=np.asarray(DEFAULT_BINS, dtype=np.float64),
        help="Comma-separated finite lower boundaries; -inf/+inf outer bins are added",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Evaluate at most this many frames across all files; 0 means all",
    )
    return parser.parse_args()


def discover_datasets(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".npy":
            raise ValueError(f"Dataset file must end in .npy: {path}")
        return [path]
    if path.is_dir():
        files = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".npy")
        if not files:
            raise FileNotFoundError(f"No .npy files directly under dataset directory: {path}")
        return files
    raise FileNotFoundError(path)


def extract_field_input(batch: np.ndarray, field: str, input_layout: str) -> np.ndarray:
    """Convert raw [quat, root vel, root acc, q, qdot, qddot] rows to model order."""
    columns = int(batch.shape[1])
    expected = FIELD_INPUT_DIMS[field]
    if input_layout == "raw" and columns != 97:
        raise ValueError(f"--input-layout raw requires 97 columns, got {columns}")
    use_raw = input_layout == "raw" or (input_layout == "auto" and columns == 97)
    if use_raw:
        root_tilt, root_velocity, root_acceleration = convert_root_features(
            batch[:, 0:4], batch[:, 4:7], batch[:, 7:10]
        )
        pose = np.concatenate((root_tilt, batch[:, 10:39]), axis=1)
        if field == "pose":
            return pose
        velocity = np.concatenate((pose, root_velocity, batch[:, 39:68]), axis=1)
        if field == "vel":
            return velocity
        return np.concatenate((velocity, root_acceleration, batch[:, 68:97]), axis=1)
    if columns == expected:
        return np.asarray(batch)
    # Also accept generated sample matrices and ignore their final target column.
    if columns in {expected + 1, expected + 2}:
        return np.asarray(batch[:, :expected])
    raise ValueError(
        f"Field {field!r} accepts raw 97-column data, {expected}-column model input, "
        f"or sample rows with one/two targets; got {columns} columns"
    )


def iter_batches(
    paths: Iterable[Path], field: str, batch_size: int, max_frames: int, input_layout: str
) -> Iterable[tuple[Path, np.ndarray]]:
    emitted = 0
    for path in paths:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 2 or array.dtype != np.float32:
            raise ValueError(f"Dataset must be a 2D float32 NPY, got {array.shape} {array.dtype}: {path}")
        for start in range(0, len(array), batch_size):
            if max_frames > 0 and emitted >= max_frames:
                return
            end = min(start + batch_size, len(array))
            if max_frames > 0:
                end = min(end, start + max_frames - emitted)
            raw = np.asarray(array[start:end])
            features = np.ascontiguousarray(
                extract_field_input(raw, field, input_layout), dtype=np.float32
            )
            emitted += len(features)
            yield path, features


def format_edge(value: float) -> str:
    if math.isinf(value):
        return "-inf" if value < 0 else "+inf"
    return f"{value:g}"


def print_report(report: dict) -> None:
    summary = report["summary"]
    print(f"field      : {report['field']}")
    print(f"checkpoint : {report['checkpoint']}")
    print(f"dataset    : {report['dataset']}")
    print(f"frames     : {summary['count']:,}")
    print(
        "distance   : "
        f"mean={summary['mean']:.8f}, std={summary['std']:.8f}, "
        f"min={summary['min']:.8f}, max={summary['max']:.8f}"
    )
    print()
    print(f"{'distance interval':>24} | {'frames':>14} | {'percent':>9}")
    print(f"{'-' * 24}-+-{'-' * 14}-+-{'-' * 9}")
    for item in report["intervals"]:
        label = f"[{item['lower']}, {item['upper']})"
        print(f"{label:>24} | {item['count']:>14,} | {item['percentage']:>8.3f}%")
    print()
    print(f"JSON written to: {report['output']}")


def main(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_frames < 0:
        raise ValueError("--max-frames cannot be negative")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    metadata = torch_load_safe(checkpoint_path, map_location="cpu")
    if not isinstance(metadata, dict) or str(metadata.get("field", "")).lower() not in FIELD_INPUT_DIMS:
        raise ValueError("Checkpoint must contain field='pose', 'vel', or 'acc'")
    field = str(metadata["field"]).lower()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu")
    model, checkpoint = load_distance_field(checkpoint_path, device=device, expected_field=field)
    model.eval()
    paths = discover_datasets(args.dataset)

    finite_edges = np.asarray(args.bins, dtype=np.float64)
    histogram_edges = np.concatenate(([-np.inf], finite_edges, [np.inf]))
    counts = np.zeros(len(histogram_edges) - 1, dtype=np.int64)
    moments = RunningMoments()
    file_counts = {str(path): 0 for path in paths}

    with torch.inference_mode():
        for path, features in iter_batches(
            paths, field, args.batch_size, args.max_frames, args.input_layout
        ):
            prediction = model(torch.from_numpy(features).to(device)).float().cpu().numpy()
            if not np.isfinite(prediction).all():
                raise RuntimeError(f"Model produced NaN or infinity while processing {path}")
            moments.update(prediction)
            counts += np.histogram(prediction, bins=histogram_edges)[0]
            file_counts[str(path)] += len(prediction)

    if moments.count == 0:
        raise ValueError("Dataset contains no frames")
    output = args.output
    if output is None:
        dataset_name = args.dataset.expanduser().resolve().stem
        output = PROJECT_ROOT / "eval" / "results" / f"{dataset_name}_{field}.json"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    intervals = []
    for lower, upper, count in zip(histogram_edges[:-1], histogram_edges[1:], counts):
        intervals.append(
            {
                "lower": format_edge(float(lower)),
                "upper": format_edge(float(upper)),
                "count": int(count),
                "percentage": 100.0 * int(count) / moments.count,
            }
        )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "field": field,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)) + 1,
        "dataset": str(args.dataset.expanduser().resolve()),
        "dataset_files": file_counts,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "input_layout": args.input_layout,
        "max_frames": int(args.max_frames),
        "summary": moments.export(),
        "interval_semantics": "left-closed, right-open: [lower, upper)",
        "finite_bin_boundaries": finite_edges.tolist(),
        "intervals": intervals,
        "output": str(output),
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print_report(report)


if __name__ == "__main__":
    main(parse_args())
