from pathlib import Path
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import numpy as np
import random

from training.data import MusdbRandomChunks
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


def seed_worker(worker_id):
    """
    Function called by DataLoader for each worker.
    Ensures each worker gets a unique random seed.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset

    worker_seed = worker_info.seed % 2 ** 32

    dataset.rng = random.Random(worker_seed)

    np.random.seed(worker_seed)

def train(
        data_root: str = "./musdb18-wav",
        data_format: str = "wav",
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
        resume: str | None = None,
        log_level: int = 0,
        loss_name: str = "si_sdr_l1",
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
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    ema = EMA(model, decay=0.999)

    # Select loss function
    loss_fn = get_loss_fn(loss_name)
    print(f"Using loss: {loss_name}")

    workdir = Path(workdir)
    ckpt_dir = workdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Optional resume
    step = 0
    epoch = 0
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
            step, epoch = load_checkpoint(ckpt_path, model, optimizer, scaler)
            print(f"Resumed from: {ckpt_path} | step={step} epoch={epoch}")

    running_loss = 0.0
    last_log_time = time.monotonic()

    if log_level == 2:
        acc_data_time = 0.0
        acc_forward_time = 0.0
        acc_backward_time = 0.0
        acc_optimizer_time = 0.0

    while step < max_steps:
        if log_level == 2:
            data_start_time = time.monotonic()

        for batch in loader:
            if use_cuda and log_level == 2:
                torch.cuda.synchronize()
            if log_level == 2:
                data_end_time = time.monotonic()
                acc_data_time += (data_end_time - data_start_time)

                forward_start_time = time.monotonic()

            mixture = batch["mixture"].to(device, non_blocking=True)  # (B, C=1, L)
            targets = batch["targets"].to(device, non_blocking=True)  # (B, S, C=1, L)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
                masks = model(mixture)  # (B, S, Lm)
                preds = apply_masks(mixture, masks)  # (B, S, 1, Lp)
                Lp = preds.shape[-1]
                targets_c = targets[..., :Lp]  # match length
                loss = loss_fn(preds, targets_c)

            if use_cuda and log_level == 2:
                torch.cuda.synchronize()
            if log_level == 2:
                forward_end_time = time.monotonic()
                acc_forward_time += (forward_end_time - forward_start_time)
                backward_start_time = time.monotonic()

            scaler.scale(loss).backward()

            if use_cuda and log_level == 2:
                torch.cuda.synchronize()
            if log_level == 2:
                backward_end_time = time.monotonic()
                acc_backward_time += (backward_end_time - backward_start_time)

                optimizer_start_time = time.monotonic()

            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            if use_cuda and log_level == 2:
                torch.cuda.synchronize()
            if log_level == 2:
                optimizer_end_time = time.monotonic()
                acc_optimizer_time += (optimizer_end_time - optimizer_start_time)

            running_loss += float(loss.item())
            step += 1

            if step % log_every == 0:
                avg_loss = running_loss / log_every

                current_time = time.monotonic()
                total_duration_wall = current_time - last_log_time
                steps_per_sec = log_every / total_duration_wall

                log_parts = []

                if log_level >= 1:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_parts.append(f"[{timestamp}]")

                log_parts.append(f"Step {step:6d}")
                log_parts.append(f"Avg Loss: {avg_loss:.5f}")
                log_parts.append(f"{steps_per_sec:.2f} steps/s")

                print(f"\n{' | '.join(log_parts)}")

                if log_level == 2:
                    avg_data_time_ms = (acc_data_time / log_every) * 1000
                    avg_forward_time_ms = (acc_forward_time / log_every) * 1000
                    avg_backward_time_ms = (acc_backward_time / log_every) * 1000
                    avg_optimizer_time_ms = (acc_optimizer_time / log_every) * 1000

                    total_measured_ms = avg_data_time_ms + avg_forward_time_ms + avg_backward_time_ms + avg_optimizer_time_ms

                    print(f"  Average times per step (total measured: {total_measured_ms:.2f} ms):")

                    if total_measured_ms > 0:
                        print(
                            f"    - Data Loading:       {avg_data_time_ms:8.2f} ms ({avg_data_time_ms / total_measured_ms * 100:5.1f}%)")
                        print(
                            f"    - Forward Pass:       {avg_forward_time_ms:8.2f} ms ({avg_forward_time_ms / total_measured_ms * 100:5.1f}%)")
                        print(
                            f"    - Backward Pass:      {avg_backward_time_ms:8.2f} ms ({avg_backward_time_ms / total_measured_ms * 100:5.1f}%)")
                        print(
                            f"    - Optimizer/Update:   {avg_optimizer_time_ms:8.2f} ms ({avg_optimizer_time_ms / total_measured_ms * 100:5.1f}%)")

                running_loss = 0.0
                last_log_time = current_time

                if log_level == 2:
                    acc_data_time = 0.0
                    acc_forward_time = 0.0
                    acc_backward_time = 0.0
                    acc_optimizer_time = 0.0

            if step % ckpt_every == 0:
                ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
                save_checkpoint(ckpt_path, model, optimizer, step, epoch, scaler)
                print(f"Saved checkpoint: {ckpt_path}")

            if step >= max_steps:
                break

            if log_level == 2:
                data_start_time = time.monotonic()
        epoch += 1

    # Final save
    ckpt_path = ckpt_dir / f"final_step_{step:06d}.pt"
    save_checkpoint(ckpt_path, model, optimizer, step, epoch, scaler)
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
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--ckpt_every", type=int, default=500)
    parser.add_argument("--segment_seconds", type=float, default=6.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help='path to checkpoint or "auto"')
    parser.add_argument("--log_level", type=int, default=0, choices=[0, 1, 2],
                        help="Logging level: 0 (minimal), 1 (+timestamp), 2 (detailed timings, worse performance)")
    parser.add_argument("--loss", type=str, default="si_sdr_l1",
                        choices=["l1", "si_sdr", "mrstft", "si_sdr_l1", "combo", "perceptual"],
                        help="Loss to use: l1, si_sdr, mrstft, si_sdr_l1 (hybrid), combo (si_sdr + 0.1*mrstft), perceptual (si_sdr + 0.5*l1 + 0.2*mrstft)")

    args = parser.parse_args()

    print(args)

    train(
        data_root=args.data_root,
        data_format=args.data_format,
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
        log_level=args.log_level,
        loss_name=args.loss,
    )
