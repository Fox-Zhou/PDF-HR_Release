from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from PDFHR.encoders import merge_strict
from PDFHR.field_specs import FEATURE_LAYOUT, FieldSpec, get_field_spec
from PDFHR.model import build_distance_field, get_default_model_cfg, torch_load_safe

from .data import (
    SampleStore,
    build_batch_refs,
    compute_feature_stats,
    discover_shards,
    expand_path,
    shuffled_rank_refs,
)


COMMON_DEFAULT_CFG: dict[str, Any] = {
    "field": "pose",
    "root": None,
    "sample_dir": "",
    "output_dir": "",
    "run_name": "distance_field",
    "resume": None,
    "save_interval": 5,
    "model": {},
    "train": {
        "epochs": 50,
        "batch_size": 8192,
        "max_files": 0,
        "validation_fraction": 0.02,
        "max_batches_per_epoch": 0,
        "stats_max_batches": 64,
        "loss": "mse",
        "optimizer": {"type": "adamw", "lr": 1.0e-3, "weight_decay": 1.0e-5},
        "scheduler": {"type": "cosine", "min_lr": 1.0e-5},
        "device": "cuda",
        "amp": True,
        "pin_memory": True,
        "random_seed": 42,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PDF-HR distance field")
    parser.add_argument("--config", required=True, help="Path to a strict JSON config")
    parser.add_argument("--resume", default=None, help="Checkpoint override for config.resume")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = expand_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        user_cfg = json.load(handle)
    if not isinstance(user_cfg, dict):
        raise TypeError("Config must be a JSON object")
    if "root" not in user_cfg or not isinstance(user_cfg["root"], bool):
        raise TypeError("Config root is required and must be true or false")
    field = str(user_cfg.get("field", "pose")).lower()
    get_field_spec(field)
    default = deepcopy(COMMON_DEFAULT_CFG)
    default["field"] = field
    default["run_name"] = f"{field}_distance_field"
    default["model"] = get_default_model_cfg(field, root=user_cfg["root"])
    cfg = merge_strict(default, user_cfg, "config")
    cfg["field"] = str(cfg["field"]).lower()
    if cfg["field"] != field:
        raise ValueError("Config field normalization failed")
    if not cfg["sample_dir"] or not cfg["output_dir"]:
        raise ValueError("sample_dir and output_dir are required")
    if not str(cfg["run_name"]).strip():
        raise ValueError("run_name must not be empty")
    if int(cfg["save_interval"]) <= 0:
        raise ValueError("save_interval must be positive")
    train = cfg["train"]
    for name in ("epochs", "batch_size"):
        if int(train[name]) <= 0:
            raise ValueError(f"train.{name} must be positive")
    for name in ("max_files", "max_batches_per_epoch", "stats_max_batches"):
        if int(train[name]) < 0:
            raise ValueError(f"train.{name} must be non-negative")
    if not 0.0 <= float(train["validation_fraction"]) < 1.0:
        raise ValueError("train.validation_fraction must be in [0, 1)")
    return cfg


def setup_distributed(device_name: str) -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    requested = str(device_name).lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("train.device='cuda' but CUDA is unavailable")
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only "
                f"{torch.cuda.device_count()} CUDA devices exist"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    elif requested == "cpu":
        device = torch.device("cpu")
        backend = "gloo"
    else:
        raise ValueError("train.device must be 'cuda' or 'cpu'")
    if world_size > 1:
        dist.init_process_group(backend=backend, init_method="env://")
    return rank, local_rank, world_size, device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(cfg["type"]).lower()
    kwargs = {"lr": float(cfg["lr"]), "weight_decay": float(cfg["weight_decay"])}
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    raise ValueError("train.optimizer.type must be 'adamw' or 'adam'")


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: dict[str, Any], epochs: int
) -> torch.optim.lr_scheduler._LRScheduler | None:
    name = str(cfg["type"]).lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=float(cfg["min_lr"])
        )
    raise ValueError("train.scheduler.type must be 'cosine' or 'none'")


def build_loss(name: str) -> nn.Module:
    normalized = str(name).lower()
    if normalized == "mse":
        return nn.MSELoss()
    if normalized == "l1":
        return nn.L1Loss()
    if normalized == "smooth_l1":
        return nn.SmoothL1Loss()
    raise ValueError("train.loss must be 'mse', 'l1', or 'smooth_l1'")


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def stats_to_vector(
    stats: dict[str, dict[str, Any]], spec: FieldSpec
) -> np.ndarray:
    values: list[np.ndarray] = []
    for group in spec.stat_groups:
        for key in ("mean", "std"):
            value = np.asarray(stats[group.name][key], dtype=np.float32)
            if value.shape != (group.width,):
                raise ValueError(
                    f"Statistics {group.name}.{key} must have width {group.width}"
                )
            values.append(value)
    return np.concatenate(values)


def vector_to_stats(vector: np.ndarray, spec: FieldSpec) -> dict[str, dict[str, np.ndarray]]:
    stats: dict[str, dict[str, np.ndarray]] = {}
    offset = 0
    for group in spec.stat_groups:
        mean = vector[offset : offset + group.width].copy()
        offset += group.width
        std = vector[offset : offset + group.width].copy()
        offset += group.width
        stats[group.name] = {"mean": mean, "std": std}
    if offset != len(vector):
        raise ValueError("Statistics vector has an unexpected length")
    return stats


def broadcast_stats(
    stats: dict[str, dict[str, Any]] | None,
    spec: FieldSpec,
    *,
    is_main: bool,
    world_size: int,
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    vector_length = 2 * sum(group.width for group in spec.stat_groups)
    tensor = torch.empty(vector_length, dtype=torch.float32, device=device)
    if is_main:
        if stats is None:
            raise ValueError("Main rank has no feature statistics")
        tensor.copy_(torch.from_numpy(stats_to_vector(stats, spec)).to(device))
    if world_size > 1:
        dist.broadcast(tensor, src=0)
    return vector_to_stats(tensor.cpu().numpy(), spec)


def checkpoint_payload(
    *,
    epoch: int,
    global_step: int,
    cfg: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    scaler: torch.cuda.amp.GradScaler,
) -> dict[str, Any]:
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    return {
        "feature_layout": FEATURE_LAYOUT,
        "field": cfg["field"],
        "root": cfg["root"],
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_cfg": deepcopy(raw_model.model_cfg),
        "data_cfg": {
            "sample_dir": str(expand_path(cfg["sample_dir"])),
            "batch_size": int(cfg["train"]["batch_size"]),
            "validation_fraction": float(cfg["train"]["validation_fraction"]),
        },
        "train_cfg": deepcopy(cfg["train"]),
        "feature_stats": raw_model.export_feature_stats(),
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }


def reduce_metrics(
    values: list[float], device: torch.device, world_size: int
) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def evaluate(
    model: nn.Module,
    store: SampleStore,
    refs: list[Any],
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float, int]:
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    count = 0
    with torch.no_grad():
        for ref in refs[rank::world_size]:
            features, target = store.load(ref)
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                prediction = raw_model(features)
            difference = prediction.float() - target
            squared_error += torch.square(difference).sum().item()
            absolute_error += torch.abs(difference).sum().item()
            count += int(target.numel())
    squared_error, absolute_error, count_float = reduce_metrics(
        [squared_error, absolute_error, count], device, world_size
    )
    count = int(count_float)
    if count == 0:
        return math.nan, math.nan, 0
    return squared_error / count, absolute_error / count, count


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.resume is not None:
        cfg["resume"] = args.resume
    field = cfg["field"]
    spec = get_field_spec(field)
    train_cfg = cfg["train"]
    rank, _local_rank, world_size, device = setup_distributed(train_cfg["device"])
    is_main = rank == 0
    try:
        seed_everything(int(train_cfg["random_seed"]))
        sample_dir = expand_path(cfg["sample_dir"])
        output_dir = expand_path(cfg["output_dir"]) / str(cfg["run_name"])
        shards = discover_shards(sample_dir, spec, int(train_cfg["max_files"]))
        train_refs, validation_refs, dropped_rows = build_batch_refs(
            shards,
            int(train_cfg["batch_size"]),
            float(train_cfg["validation_fraction"]),
        )
        store = SampleStore(shards, spec, root=cfg["root"])

        resume_checkpoint: dict[str, Any] | None = None
        if cfg["resume"]:
            resume_checkpoint = torch_load_safe(
                expand_path(cfg["resume"]), map_location="cpu"
            )
            if (
                not isinstance(resume_checkpoint, dict)
                or resume_checkpoint.get("feature_layout") != FEATURE_LAYOUT
                or resume_checkpoint.get("field") != field
                or resume_checkpoint.get("root") != cfg["root"]
            ):
                raise ValueError(
                    f"resume must point to a {FEATURE_LAYOUT} PDF-HR {field} "
                    f"checkpoint with root={cfg['root']}"
                )
            configured_model = merge_strict(
                get_default_model_cfg(field, root=cfg["root"]),
                cfg["model"],
                "model",
            )
            if resume_checkpoint["model_cfg"] != configured_model:
                raise ValueError(
                    "Resume checkpoint model_cfg does not match the current config"
                )
            feature_stats = resume_checkpoint["feature_stats"]
            stats_rows = 0
        else:
            rank_stats = None
            stats_rows = 0
            if is_main:
                rank_stats, stats_rows = compute_feature_stats(
                    store, train_refs, int(train_cfg["stats_max_batches"])
                )
            feature_stats = broadcast_stats(
                rank_stats,
                spec,
                is_main=is_main,
                world_size=world_size,
                device=device,
            )

        model = build_distance_field(
            field,
            cfg["model"],
            feature_stats,
            device=device,
            root=cfg["root"],
        )
        if world_size > 1:
            ddp_kwargs = {"device_ids": [device.index]} if device.type == "cuda" else {}
            model = DistributedDataParallel(model, **ddp_kwargs)
        optimizer = build_optimizer(model, train_cfg["optimizer"])
        scheduler = build_scheduler(
            optimizer, train_cfg["scheduler"], int(train_cfg["epochs"])
        )
        amp_enabled = bool(train_cfg["amp"]) and device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        criterion = build_loss(train_cfg["loss"])
        start_epoch = 0
        global_step = 0
        if resume_checkpoint is not None:
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model.load_state_dict(resume_checkpoint["model"], strict=True)
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
            if scheduler is None and resume_checkpoint["scheduler"] is not None:
                raise ValueError("Checkpoint uses a scheduler but config disables it")
            if scheduler is not None:
                if resume_checkpoint["scheduler"] is None:
                    raise ValueError("Config uses a scheduler but checkpoint does not")
                scheduler.load_state_dict(resume_checkpoint["scheduler"])
            scaler.load_state_dict(resume_checkpoint.get("scaler", {}))
            start_epoch = int(resume_checkpoint["epoch"]) + 1
            global_step = int(resume_checkpoint["global_step"])

        if is_main:
            parameters = sum(parameter.numel() for parameter in model.parameters())
            print(
                f"{field} root={cfg['root']} | shards={len(shards)} "
                f"train_batches={len(train_refs)} "
                f"val_batches={len(validation_refs)} dropped_tail_rows={dropped_rows}"
            )
            print(
                f"world_size={world_size} batch_size_per_rank={train_cfg['batch_size']} "
                f"parameters={parameters:,} amp={amp_enabled} stats_rows={stats_rows}"
            )
        if start_epoch >= int(train_cfg["epochs"]):
            if is_main:
                print("Checkpoint has already reached the configured epoch count; nothing to do.")
            return

        pin_memory = bool(train_cfg["pin_memory"]) and device.type == "cuda"
        for epoch in range(start_epoch, int(train_cfg["epochs"])):
            epoch_start = time.monotonic()
            epoch_refs, global_batch_count = shuffled_rank_refs(
                train_refs,
                seed=int(train_cfg["random_seed"]),
                epoch=epoch,
                rank=rank,
                world_size=world_size,
                max_batches_per_epoch=int(train_cfg["max_batches_per_epoch"]),
            )
            model.train()
            loss_sum = 0.0
            absolute_error = 0.0
            count = 0
            for ref in epoch_refs:
                features, target = store.load(ref)
                if pin_memory:
                    features = features.pin_memory()
                    target = target.pin_memory()
                features = features.to(device, non_blocking=pin_memory)
                target = target.to(device, non_blocking=pin_memory)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    prediction = model(features)
                    loss = criterion(prediction, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                batch_count = int(target.numel())
                loss_sum += float(loss.detach()) * batch_count
                absolute_error += torch.abs(
                    prediction.detach().float() - target
                ).sum().item()
                count += batch_count
                global_step += 1
            loss_sum, absolute_error, count_float = reduce_metrics(
                [loss_sum, absolute_error, count], device, world_size
            )
            count = int(count_float)
            train_loss = loss_sum / count
            train_mae = absolute_error / count
            val_mse, val_mae, val_count = evaluate(
                model,
                store,
                validation_refs,
                rank=rank,
                world_size=world_size,
                device=device,
                amp_enabled=amp_enabled,
            )
            if scheduler is not None:
                scheduler.step()
            if is_main:
                elapsed = time.monotonic() - epoch_start
                print(
                    f"epoch {epoch + 1}/{train_cfg['epochs']} | "
                    f"train_loss={train_loss:.8f} train_mae={train_mae:.8f} | "
                    f"val_mse={val_mse:.8f} val_mae={val_mae:.8f} "
                    f"val_rows={val_count} batches={global_batch_count} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e} time={elapsed:.1f}s"
                )
                if (
                    (epoch + 1) % int(cfg["save_interval"]) == 0
                    or epoch + 1 == int(train_cfg["epochs"])
                ):
                    payload = checkpoint_payload(
                        epoch=epoch,
                        global_step=global_step,
                        cfg=cfg,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                    )
                    epoch_path = output_dir / f"epoch_{epoch + 1:04d}.pt"
                    atomic_torch_save(payload, epoch_path)
                    atomic_torch_save(payload, output_dir / "latest.pt")
                    print(f"saved {epoch_path}")
        store.close()
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
