from pathlib import Path
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import numpy as np
import random

from models.unet2d.model import Unet2DWrapper
from models.unet2d.train_step import train_step
from training.data import MusdbRandomChunks
from models.unet2d.eval import evaluate as eval_unet
from training.utils import load_checkpoint, save_checkpoint, find_latest_checkpoint
from training.losses import get_loss_fn

torch.backends.cudnn.benchmark = True

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


def log_status(epoch, running_loss, log_every, epoch_size, last_log_time):
    avg_loss = running_loss / (log_every * epoch_size)

    current_time = time.monotonic()
    total_duration_wall = current_time - last_log_time
    steps_per_sec = (log_every * epoch_size) / total_duration_wall

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] | Epoch {epoch:6d} | Avg Loss: {avg_loss:.5f} | {steps_per_sec:.2f} steps/s")

    return current_time


def run_eval(epoch, model, device, data_root, sources, segment_seconds, batch_size, num_workers):
    model.eval()

    metrics = eval_unet(
        model,
        data_root=data_root,
        sources=list(sources),
        segment_seconds=segment_seconds,
        device=device,
        subset="test",
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model.train()

    if metrics:
        print(f"[Eval] Epoch {epoch:6d} | Metrics:")
        for key in metrics.keys():
            print(f"    {key:16s}: {metrics[key]:8.3f}")
    else:
        print(f"[Eval] Epoch {epoch:6d} | No metrics returned from eval_unet")

    return metrics


def prune_previous_checkpoint(ckpt_dir: Path, current_epoch: int, ckpt_every: int):
    target_epoch = current_epoch - ckpt_every
    if target_epoch <= 0:
        return
    old_path = ckpt_dir / f"epoch_{target_epoch:04d}.pt"
    if old_path.exists():
        old_path.unlink()
        print(f"[CKPT] Removed old checkpoint: {old_path}")


def train(
        data_root: str = "./musdb18-wav",
        workdir: str = "./runs/unet1d",
        batch_size: int = 4,
        lr: float = 2e-4,
        epochs: int = 400,
        epoch_size: int = 50,
        log_every: int = 1,
        ckpt_every: int = 10,
        segment_seconds: float = 6.0,
        num_workers: int = 4,
        sources: list[str] = ["vocals", "drums", "bass", "other"],
        seed: int = 42,
        resume: str | None = None,
        loss_name: str = "si_sdr_l1",
        eval_every: int = 5,
        prune_checkpoints: bool = False,
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
        subset="train",
        sources=sources,
        segment_seconds=segment_seconds,
        items_per_epoch=10_000,
        mono=False,
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

    model = Unet2DWrapper(stem_names=sources).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

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

    epoch = 0
    for epoch in range(start_epoch + 1, epochs + 1):
        for i, batch in enumerate(loader, start=1):
            step += 1

            loss = train_step(model, batch, optimizer, scaler, loss_fn, device, use_cuda)
            running_loss += loss

            if i >= epoch_size:
                break

        if epoch % log_every == 0:
            last_log_time = log_status(epoch, running_loss, log_every, epoch_size, last_log_time)
            running_loss = 0.0

        if epoch % eval_every == 0:
            metrics = run_eval(epoch, model, device, data_root, sources, segment_seconds, batch_size, num_workers)
            current_si_sdr = metrics.get("si_sdr/mean", float("-inf"))            

        if epoch % ckpt_every == 0:
            path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            try:
                save_checkpoint(path, model, optimizer, epoch, step, scaler)
            except Exception as exc:
                print(f"[CKPT] Failed to save checkpoint {path}: {exc}")
                raise
            else:
                print(f"[CKPT] Saved checkpoint: {path}")
                if prune_checkpoints and path.exists():
                    prune_previous_checkpoint(ckpt_dir, epoch, ckpt_every)

    # Final save
    ckpt_path = ckpt_dir / f"final_step_{step:06d}.pt"
    save_checkpoint(ckpt_path, model, optimizer, epoch, step, scaler)
    print(f"Saved final checkpoint: {ckpt_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a 1D U-Net for audio source separation (MUSDB18)")
    parser.add_argument("--data_root", type=str, default="./musdb18-wav")
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
                        choices=["l1", "l2", "si_sdr", "mrstft", "si_sdr_l1"],
                        help="Loss to use: l1, l2, si_sdr, mrstft, si_sdr_l1 (hybrid)")
    parser.add_argument("--eval_every", type=int, default=5, help="Evaluate every N epochs")
    parser.add_argument("--prune_checkpoints", action="store_true",
                        help="Delete the previous scheduled checkpoint each time a new one is saved")

    args = parser.parse_args()

    print(args)

    train(
        data_root=args.data_root,
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
        prune_checkpoints=args.prune_checkpoints,
    )
