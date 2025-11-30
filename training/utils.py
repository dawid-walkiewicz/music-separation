from pathlib import Path
from typing import Dict

import torch


def save_checkpoint(
        path: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        scaler: torch.cuda.amp.GradScaler | None,
        extra: Dict | None = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "extra": extra or {},
        },
        str(path),
    )


def load_checkpoint(path: Path,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    scaler: torch.cuda.amp.GradScaler | None = None):
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"]:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    epoch = int(ckpt.get("epoch", 0))
    step = int(ckpt.get("step", 0))
    return epoch, step


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    if ckpts:
        return ckpts[-1]
    finals = sorted(ckpt_dir.glob("final_step_*.pt"))
    return finals[-1] if finals else None
