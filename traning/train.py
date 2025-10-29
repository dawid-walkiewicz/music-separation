import os
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from traning.data import MusdbRandomChunks
from traning.model import UNet1D, apply_masks


class EMA:
    """Prosty Exponential Moving Average dla stabilizacji ewaluacji."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    scaler: torch.cuda.amp.GradScaler | None,
    extra: Dict | None = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "extra": extra or {},
        },
        str(path),
    )


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None,
                    scaler: torch.cuda.amp.GradScaler | None = None):
    ckpt = torch.load(str(path), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"]:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    step = int(ckpt.get("step", 0))
    epoch = int(ckpt.get("epoch", 0))
    return step, epoch


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    if ckpts:
        return ckpts[-1]
    finals = sorted(ckpt_dir.glob("final_step_*.pt"))
    return finals[-1] if finals else None


def train(
    data_root: str = "./musdb18",
    workdir: str = "./runs/unet1d",
    batch_size: int = 4,
    lr: float = 2e-4,
    max_steps: int = 10_000,
    log_every: int = 50,
    ckpt_every: int = 500,
    segment_seconds: float = 6.0,
    num_workers: int = 4,
    sources: tuple[str, ...] = ("vocals", "drums", "bass", "other"),
    seed: int = 42,
    resume: str | None = None,  # ścieżka do .pt lub "auto"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "CUDA device"
        print(f"Device: CUDA ({gpu_name}) | CUDA available: {torch.cuda.is_available()} | AMP: True")
    else:
        print("Device: CPU | AMP: False")

    dataset = MusdbRandomChunks(
        root=data_root,
        subset="train",
        sources=list(sources),
        segment_seconds=segment_seconds,
        items_per_epoch=10_000,
        mono=True,
        seed=seed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=(device.type == "cuda"))

    model = UNet1D(n_sources=len(sources), base=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    ema = EMA(model, decay=0.999)

    workdir = Path(workdir)
    ckpt_dir = workdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Opcjonalne wznowienie
    step = 0
    epoch = 0
    if resume:
        ckpt_path: Path | None
        if resume == "auto":
            ckpt_path = find_latest_checkpoint(ckpt_dir)
            if ckpt_path is None:
                print("Brak checkpointów do auto-wznowienia — start od zera.")
        else:
            ckpt_path = Path(resume)
            if not ckpt_path.exists():
                print(f"Ostrzeżenie: nie znaleziono {ckpt_path}, start od zera.")
                ckpt_path = None
        if ckpt_path is not None:
            step, epoch = load_checkpoint(ckpt_path, model, optimizer, scaler)
            print(f"Wznowiono z: {ckpt_path} | step={step} epoch={epoch}")

    running_loss = 0.0

    while step < max_steps:
        for batch in loader:
            mixture = batch["mixture"].to(device)  # (B, C=1, L)
            targets = batch["targets"].to(device)  # (B, S, C=1, L)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                masks = model(mixture)                 # (B, S, Lm)
                preds = apply_masks(mixture, masks)    # (B, S, 1, Lp)
                Lp = preds.shape[-1]
                targets_c = targets[..., :Lp]          # dopasuj długość
                loss = (preds - targets_c).abs().mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            running_loss += float(loss.item())
            step += 1

            if step % log_every == 0:
                avg = running_loss / log_every
                print(f"step {step:6d} | loss {avg:.5f}")
                running_loss = 0.0

            if step % ckpt_every == 0:
                ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
                save_checkpoint(ckpt_path, model, optimizer, step, epoch, scaler)
                print(f"Zapisano checkpoint: {ckpt_path}")

            if step >= max_steps:
                break
        epoch += 1

    # Zapis końcowy
    ckpt_path = ckpt_dir / f"final_step_{step:06d}.pt"
    save_checkpoint(ckpt_path, model, optimizer, step, epoch, scaler)
    print(f"Zapisano checkpoint końcowy: {ckpt_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trenowanie 1D U-Net do separacji źródeł audio (MUSDB18)")
    parser.add_argument("--data_root", type=str, default="./musdb18")
    parser.add_argument("--workdir", type=str, default="./runs/unet1d")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--ckpt_every", type=int, default=500)
    parser.add_argument("--segment_seconds", type=float, default=6.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help='ścieżka do ckpt lub "auto"')

    args = parser.parse_args()

    train(
        data_root=args.data_root,
        workdir=args.workdir,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        segment_seconds=args.segment_seconds,
        num_workers=args.num_workers,
        seed=args.seed,
        resume=args.resume,
    )
