#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from girnet.convolutive import ConvolutiveRenderDataset
from girnet.data import MUSDBBleedDataset
from girnet.losses import GIRNetLoss, LossWeights, ordered_si_sdr
from girnet.model import GIRNetSeparator, ModelConfig
from girnet.utils import count_parameters, file_sha256, load_config, save_json, seed_everything


def make_dataset(data_config: dict, split: str):
    if data_config.get("kind", "instantaneous") == "convolutive":
        return ConvolutiveRenderDataset(
            manifest=data_config[f"{split}_manifest"],
            stems=data_config["stems"],
            sample_rate=data_config["sample_rate"],
            segment_samples=data_config["segment_samples"],
            peak=data_config.get("peak", 0.98),
            include_contributions=False,
        )
    return MUSDBBleedDataset(
        manifest=data_config[f"{split}_manifest"],
        stems=data_config["stems"],
        sample_rate=data_config["sample_rate"],
        segment_samples=data_config["segment_samples"],
        train_bleed_levels_db=data_config["train_bleed_levels_db"],
        train_edge_jitter_db=data_config.get("train_edge_jitter_db", 0.0),
        peak=data_config.get("peak", 0.98),
    )


def run_epoch(
    model, loader, criterion, device, optimizer=None, scaler=None,
    grad_clip=0.0, accumulation_steps=1,
    amp_enabled=False, amp_dtype=torch.float16,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    examples = 0
    accumulation_steps = max(1, int(accumulation_steps))
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader):
        mixture = batch["mixture"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        matrix = batch["bleed_matrix"].to(device, non_blocking=True)
        batch_size = mixture.shape[0]
        with torch.amp.autocast(
            device_type=device.type,
            enabled=bool(amp_enabled),
            dtype=amp_dtype,
        ):
            output = model(mixture)
        with torch.amp.autocast(device_type=device.type, enabled=False):
            loss, terms = criterion(
                output,
                target,
                matrix,
                mixture=mixture,
                bleed_level_db=batch["bleed_level_db"],
            )
        if not torch.isfinite(loss):
            details = {name: float(value.detach()) for name, value in terms.items()}
            raise FloatingPointError(f"Non-finite loss detected before backward: {details}")
        if training:
            scaler.scale(loss / accumulation_steps).backward()
            should_step = (batch_index + 1) % accumulation_steps == 0 or (batch_index + 1) == len(loader)
            if should_step:
                scaler.unscale_(optimizer)
                bad_gradients = [
                    name for name, parameter in model.named_parameters()
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                ]
                if bad_gradients:
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        "Non-finite gradients detected; refusing optimizer step. "
                        f"First affected parameters: {bad_gradients[:10]}"
                    )
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            metrics = {name: float(value.detach()) for name, value in terms.items()}
            metrics["si_sdr_db"] = float(ordered_si_sdr(target, output["estimate"]).mean())
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value * batch_size
        examples += batch_size
    return {name: value / max(examples, 1) for name, value in totals.items()}


def write_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    device = torch.device(args.device)
    train_config = config["training"]
    output_dir = Path(train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "resolved_config.json", config)

    train_dataset = make_dataset(config["data"], "train")
    val_dataset = make_dataset(config["data"], "val")
    loader_options = dict(
        batch_size=int(train_config["batch_size"]),
        num_workers=int(train_config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    generator = torch.Generator().manual_seed(int(config["seed"]))
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, drop_last=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_options)

    model_config = ModelConfig(**config["model"])
    model = GIRNetSeparator(model_config).to(device)
    initial_checkpoint = train_config.get("initial_checkpoint")
    initialization = None
    if initial_checkpoint:
        initial_path = Path(initial_checkpoint)
        saved = torch.load(initial_path, map_location=device, weights_only=False)
        saved_config = dict(saved.get("model_config", {}))
        if saved_config and saved_config != model.export_config():
            raise ValueError(
                f"Initial checkpoint model configuration {saved_config} does not match "
                f"requested configuration {model.export_config()}"
            )
        model.load_state_dict(saved["model"] if "model" in saved else saved)
        initialization = {"path": str(initial_path.resolve()), "sha256": file_sha256(initial_path)}
    loss_config = dict(config["loss"])
    loss_config["fft_sizes"] = tuple(loss_config.get("fft_sizes", (512, 1024, 2048)))
    criterion = GIRNetLoss(LossWeights(**loss_config), graph_mode=model_config.mode == "girnet").to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    amp_enabled = bool(train_config.get("amp", True)) and device.type == "cuda"
    amp_name = str(train_config.get("amp_dtype", "bfloat16")).lower()
    amp_types = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}
    if amp_name not in amp_types:
        raise ValueError(f"Unknown training.amp_dtype={amp_name}; use bfloat16 or float16")
    amp_dtype = amp_types[amp_name]
    # BF16 has float32-like exponent range and does not need loss scaling.
    # GradScaler remains available only for explicitly requested FP16 runs.
    scaler_enabled = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    start_epoch, best_value = 0, float("inf")
    history = read_history(output_dir / "history.csv") if args.resume else []
    bad_epochs = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["model_config"] != model.export_config():
            raise ValueError("Checkpoint model configuration does not match current configuration")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_value = float(checkpoint.get("best_val_loss", best_value))
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            # Backward-compatible recovery for checkpoints written before scheduler
            # state was persisted.
            for row in history:
                scheduler.step(float(row["val_total"]))
        if "loader_generator_state" in checkpoint:
            generator.set_state(checkpoint["loader_generator_state"])
        if "bad_epochs" in checkpoint:
            bad_epochs = int(checkpoint["bad_epochs"])
        elif history:
            best_index = min(range(len(history)), key=lambda i: float(history[i]["val_total"]))
            bad_epochs = len(history) - best_index - 1
        if history and int(history[-1]["epoch"]) != int(checkpoint["epoch"]):
            raise ValueError(
                f"History/checkpoint mismatch: history ends at {history[-1]['epoch']}, "
                f"checkpoint is epoch {checkpoint['epoch']}"
            )

    metadata = {
        "parameters": count_parameters(model),
        "device": str(device),
        "torch_version": torch.__version__,
        "amp": {"enabled": amp_enabled, "dtype": amp_name, "grad_scaler": scaler_enabled},
        "model_config": model.export_config(),
        "initial_checkpoint": initialization,
        "manifest_sha256": {
            "train": file_sha256(config["data"]["train_manifest"]),
            "val": file_sha256(config["data"]["val_manifest"]),
        },
    }
    save_json(output_dir / "run_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))

    patience = int(train_config.get("patience", 20))
    for epoch in range(start_epoch, int(train_config["epochs"])):
        train_dataset.set_epoch(epoch)
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler,
            grad_clip=float(train_config.get("grad_clip", 0.0)),
            accumulation_steps=int(train_config.get("accumulation_steps", 1)),
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model, val_loader, criterion, device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
        scheduler.step(val_metrics["total"])
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        write_history(output_dir / "history.csv", history)
        improved = val_metrics["total"] < best_value
        if improved:
            best_value = val_metrics["total"]
            bad_epochs = 0
        else:
            bad_epochs += 1
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "loader_generator_state": generator.get_state(),
            "bad_epochs": bad_epochs,
            "model_config": model.export_config(),
            "best_val_loss": best_value,
            "config": config,
        }
        torch.save(checkpoint, output_dir / "latest.pt")
        if improved:
            torch.save(checkpoint, output_dir / "best.pt")
        print(json.dumps(row, sort_keys=True))
        if bad_epochs >= patience:
            print(f"Early stopping after {bad_epochs} epochs without validation improvement")
            break


if __name__ == "__main__":
    main()
