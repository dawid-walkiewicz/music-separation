from functools import reduce
from pathlib import Path
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import numpy as np
import random

from training.data import MusdbRandomChunks
from training.evaluate import evaluate
from training.model import UNet1D, apply_masks
from training.utils import load_checkpoint, save_checkpoint, find_latest_checkpoint
from training.losses import get_loss_fn

torch.backends.cudnn.benchmark = True


class EMA:
    """Simple Exponential Moving Average for stabilizing evaluation."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

        self.buffers = {name: buf.detach().clone() for name, buf in model.named_buffers()}

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

        for name, buf in model.named_buffers():
            if name in self.buffers:
                buf.copy_(self.buffers[name])


def seed_worker(worker_id):
    """
    Function called by DataLoader for each worker.
    Ensures each worker gets a unique random seed.
    """
    worker_info = torch.utils.data.get_worker_info()
    worker_seed = worker_info.seed % 2 ** 32

    dataset = worker_info.dataset

    random.seed(worker_seed)
    dataset.rng = random.Random(worker_seed)
    np.random.default_rng(worker_seed)
    torch.manual_seed(worker_seed)


def train_step(model, batch, optimizer, scaler, ema, loss_fn, device, use_cuda):
    mixture = batch["mixture"].to(device, non_blocking=True)  # (B, C=1, L)
    targets = batch["targets"].to(device, non_blocking=True)  # (B, S, C=1, L)

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
        masks = model(mixture)  # (B, S, Lm)
        preds = apply_masks(mixture, masks)  # (B, S, 1, Lp)
        Lp = preds.shape[-1]
        targets_c = targets[..., :Lp]  # match length
        loss = loss_fn(preds, targets_c)

    scaler.scale(loss).backward()

    scaler.step(optimizer)
    scaler.update()
    ema.update(model)

    return float(loss.item())


def log_status(epoch, running_loss, log_every , epoch_size, last_log_time):
    avg_loss = running_loss / (log_every * epoch_size)

    current_time = time.monotonic()
    total_duration_wall = current_time - last_log_time
    steps_per_sec = (log_every * epoch_size) / total_duration_wall

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] | Epoch {epoch:6d} | Avg Loss: {avg_loss:.5f} | {steps_per_sec:.2f} steps/s")

    return current_time


def run_eval(epoch, model, ema, device, data_root, data_format, sources, segment_seconds):
    print("\n[Eval] Copying EMA weights and computing metrics on the validation set...")

    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema.copy_to(model)
    metrics = evaluate(
        model,
        data_root=data_root,
        data_format=data_format,
        sources=sources,
        segment_seconds=segment_seconds,
        device=device,
        max_batches=20,
    )
    model.load_state_dict(backup)
    print(f"[Eval] Epoch {epoch:6d} | SDR: {metrics['sdr']:.2f} dB | SI-SDR: {metrics['si_sdr']:.2f} dB")
    return metrics


def train(
        data_root: str = "./musdb18-wav",
        data_format: str = "wav",
        workdir: str = "./runs/unet1d",
        batch_size: int = 4,
        lr: float = 2e-4,
        epochs: int = 400,
        epoch_size: int = 50,
        log_every: int = 1,
        ckpt_every: int = 10,
        segment_seconds: float = 6.0,
        num_workers: int = 4,
        sources: tuple[str, ...] = ("vocals", "drums", "bass", "other"),
        seed: int = 42,
        resume: str | None = None,
        loss_name: str = "si_sdr_l1",
        eval_every: int = 5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    if use_cuda:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "CUDA device"
        print(f"Device: CUDA ({gpu_name}) | CUDA available: {torch.cuda.is_available()} | AMP: True")
    else:
        print("Device: CPU | AMP: False")

    dataset = MusdbRandomChunks(
        root=data_root,
        data_format=data_format,
        subset="train",
        sources=list(sources),
        segment_seconds=segment_seconds,
        items_per_epoch=10_000,
        mono=True,
        seed=seed,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
        worker_init_fn=seed_worker,
        generator=g,
    )

    model = UNet1D(n_sources=len(sources), base=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    ema = EMA(model, decay=0.999)

    loss_fn = get_loss_fn(loss_name)
    print(f"Using loss: {loss_name}")

    workdir = Path(workdir)
    ckpt_dir = workdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Optional resume
    start_epoch = 0
    step = 0
    if resume:
        ckpt_path: Path | None
        if resume == "auto":
            ckpt_path = find_latest_checkpoint(ckpt_dir)
            if ckpt_path is None:
                print("No checkpoints found for auto-resume — starting from scratch.")
        else:
            ckpt_path = Path(resume)
            if not ckpt_path.exists():
                print(f"Warning: {ckpt_path} not found, starting from scratch.")
                ckpt_path = None
        if ckpt_path is not None:
            start_epoch, step = load_checkpoint(ckpt_path, model, optimizer, scaler)
            print(f"Resumed from: {ckpt_path} | epoch={start_epoch}")

    running_loss = 0.0
    best_si_sdr = -float("inf")

    last_log_time = time.monotonic()

    for epoch in range(start_epoch + 1, epochs + 1):
        for i, batch in enumerate(loader, start=1):
            step += 1

            loss = train_step(model, batch, optimizer, scaler, ema, loss_fn, device, use_cuda)
            running_loss += loss

            if i >= epoch_size:
                break

        if epoch % log_every == 0:
            last_log_time = log_status(epoch, running_loss, log_every, epoch_size, last_log_time)
            running_loss = 0.0

        if epoch % eval_every == 0:
            metrics = run_eval(epoch, model, ema, device, data_root, data_format, sources, segment_seconds)
            if metrics["si_sdr"] > best_si_sdr:
                best_si_sdr = metrics["si_sdr"]
                best_ckpt = ckpt_dir / "best_sisdr.pt"
                save_checkpoint(best_ckpt, model, optimizer, step, epoch, scaler)
                print(f"[Eval] New best model (SI-SDR={best_si_sdr:.2f} dB) has been saved to {best_ckpt}")

        if epoch % ckpt_every == 0:
            path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(path, model, optimizer, epoch, step, scaler)
            print(f"[CKPT] Saved checkpoint: {path}")

    # Final save
    ckpt_path = ckpt_dir / f"final_step_{step:06d}.pt"
    save_checkpoint(ckpt_path, model, optimizer, epoch, step, scaler)
    print(f"Saved final checkpoint: {ckpt_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a 1D U-Net for audio source separation (MUSDB18)")
    parser.add_argument("--data_root", type=str, default="./musdb18-wav")
    parser.add_argument(
        "--data_format",
        type=str,
        default="wav",
        choices=["wav", "stem"],
        help="Input data format for MUSDB18: 'wav' (default) or 'stem'.",
    )
    parser.add_argument("--workdir", type=str, default="./runs/unet1d")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=400, help="Number of epochs to train")
    parser.add_argument("--epoch_size", type=int, default=50, help="Number of steps per epoch")
    parser.add_argument("--log_every", type=int, default=1, help="Log every N epochs")
    parser.add_argument("--ckpt_every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--segment_seconds", type=float, default=6.0, help="Segment length in seconds")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help='path to checkpoint or "auto"')
    parser.add_argument("--loss", type=str, default="si_sdr_l1",
                        choices=["l1", "si_sdr", "mrstft", "si_sdr_l1"],
                        help="Loss to use: l1, si_sdr, mrstft, si_sdr_l1 (hybrid)")
    parser.add_argument("--eval_every", type=int, default=5, help="Evaluate every N epochs")

    args = parser.parse_args()

    print(args)

    train(
        data_root=args.data_root,
        data_format=args.data_format,
        workdir=args.workdir,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        epoch_size=args.epoch_size,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        segment_seconds=args.segment_seconds,
        num_workers=args.num_workers,
        seed=args.seed,
        resume=args.resume,
        loss_name=args.loss,
        eval_every=args.eval_every,
    )
