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
from training.model import HybridTimeFreqUNet, reconstruct_sources, similarity_percent
from training.utils import load_checkpoint, save_checkpoint, find_latest_checkpoint
from training.losses import si_sdr_loss as base_sisdr_loss, mrstft_loss as mrstft_loss_freq, apply_pit


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

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow": {k: v.detach().clone() for k, v in self.shadow.items()},
            "buffers": {k: v.detach().clone() for k, v in self.buffers.items()},
        }

    def load_state_dict(self, state: dict | None):
        if not state:
            return
        self.decay = state.get("decay", self.decay)
        shadow = state.get("shadow", {})
        buffers = state.get("buffers", {})
        for name, tensor in shadow.items():
            if name in self.shadow:
                self.shadow[name].copy_(tensor)
        for name, tensor in buffers.items():
            if name in self.buffers:
                self.buffers[name].copy_(tensor)


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


def train_step(model, batch, optimizer, scaler, ema, device, use_cuda, log_level, mrstft_chunk_size, loss_type: str):
    times = {}

    if use_cuda and log_level == 2:
        torch.cuda.synchronize()

    if log_level == 2:
        start = time.monotonic()

    mixture = batch["mixture"].to(device, non_blocking=True)  # (B, 1, L)
    targets = batch["targets"].to(device, non_blocking=True)  # (B, S, L)

    if log_level == 2:
        times["data"] = time.monotonic() - start
        start = time.monotonic()

    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_cuda):
        masks = model(mixture)  # (B, S, Lm)
        preds = reconstruct_sources(mixture, masks)  # (B, S, Lp)
        Lp = preds.shape[-1]
        targets_c = targets[..., :Lp]  # (B, S, Lp)

        # Wybór funkcji straty
        if loss_type == "si_sdr":
            loss = base_sisdr_loss(preds, targets_c)
        elif loss_type == "si_sdr_pit":
            loss = apply_pit(base_sisdr_loss, preds, targets_c)
        elif loss_type == "si_sdr_pit_mrstft":
            pit = apply_pit(base_sisdr_loss, preds, targets_c)
            try:
                stft = mrstft_loss_freq(preds, targets_c)
            except ImportError:
                stft = torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
            loss = pit + 0.1 * stft
        else:
            # Domyślnie zachowanie jak poprzednio (PIT SI-SDR + prosty MRSTFT z model.py)
            from training.model import pit_si_sdr_loss, mrstft_loss as mrstft_loss_time
            pit = pit_si_sdr_loss(preds, targets_c)
            stft = mrstft_loss_time(preds, targets_c, chunk_size=mrstft_chunk_size)
            loss = pit + 0.1 * stft

    with torch.no_grad():
        sim = similarity_percent(preds.detach(), targets_c.detach())

    if use_cuda and log_level == 2:
        torch.cuda.synchronize()

    if log_level == 2:
        times["forward"] = time.monotonic() - start
        start = time.monotonic()

    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

    if use_cuda and log_level == 2:
        torch.cuda.synchronize()

    if log_level == 2:
        times["backward"] = time.monotonic() - start
        start = time.monotonic()

    scaler.step(optimizer)
    scaler.update()
    ema.update(model)

    if use_cuda and log_level == 2:
        torch.cuda.synchronize()

    if log_level == 2:
        times["optim"] = time.monotonic() - start

    return float(loss.item()), float(sim), times


def log_status(epoch, running_loss, running_similarity, log_times, log_every, log_level, epoch_size, last_log_time):
    avg_loss = running_loss / (log_every * epoch_size)
    avg_sim = running_similarity / (log_every * epoch_size)

    current_time = time.monotonic()
    total_duration_wall = current_time - last_log_time
    steps_per_sec = (log_every * epoch_size) / total_duration_wall

    log_parts = []

    if log_level >= 1:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_parts.append(f"[{timestamp}]")

    log_parts.append(f"Epoch {epoch:6d}")
    log_parts.append(f"Avg Loss: {avg_loss:.5f}")
    log_parts.append(f"Avg Similarity: {avg_sim:5.2f}%")
    log_parts.append(f"{steps_per_sec:.2f} steps/s")

    print(f"\n{' | '.join(log_parts)}")

    if log_level == 2:
        total_measured_ms = reduce(lambda a, b: a + b,
                                      [val / log_every * 1000 for val in log_times.values()])
        print(f"  Average times per step (total measured: {total_measured_ms:.2f} ms):")

        if total_measured_ms > 0:
            for key, val in log_times.items():
                print(f"    - {key.capitalize():10s}: {val / log_every * 1000:8.2f} ms")

    return current_time


def run_eval(epoch, model, ema, device, data_root, data_format, sources, segment_seconds, max_batches=20, eval_num_workers=0):
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
        max_batches=max_batches,
        num_workers=eval_num_workers,
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
        log_level: int = 0,
        eval_every: int = 5,
        items_per_epoch: int = 10_000,
        eval_batches: int = 20,
        use_checkpoint: bool = False,
        eval_num_workers: int = 0,
        base_time: int = 128,
        base_freq: int = 64,
        mrstft_chunk_size: int = 2,
        model_type: str = "hybrid",
        loss_type: str = "si_sdr_pit",
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

    # Dataset & loader
    dataset = MusdbRandomChunks(
        root=data_root,
        data_format=data_format,
        subset="train",
        sources=list(sources),
        segment_seconds=segment_seconds,
        items_per_epoch=items_per_epoch,
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

    # Model, optimizer, EMA
    if model_type == "unet1d":
        from training.model import UNet1D
        model = UNet1D(
            n_sources=len(sources),
            base=base_time,
        ).to(device)
    else:
        model = HybridTimeFreqUNet(
            n_sources=len(sources),
            base_time=base_time,
            base_freq=base_freq,
            use_checkpoint=use_checkpoint,
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=use_cuda)
    ema = EMA(model, decay=0.999)

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
            start_epoch, step, ema_state, extra = load_checkpoint(ckpt_path, model, optimizer, scaler)
            ema.load_state_dict(ema_state)
            print(f"Resumed from: {ckpt_path} | epoch={start_epoch}")
    # Try to initialize best_si_sdr from existing best_sisdr.pt if present
    best_ckpt_path = ckpt_dir / "best_sisdr.pt"
    if best_ckpt_path.exists():
        try:
            _, _, _, best_extra = load_checkpoint(best_ckpt_path, model, None, None)
            if isinstance(best_extra, dict):
                m_best = best_extra.get("metrics")
                if m_best and "si_sdr" in m_best:
                    best_si_sdr = float(m_best["si_sdr"])
                    print(f"Loaded best SI-SDR={best_si_sdr:.2f} dB from {best_ckpt_path}")
        except Exception as exc:
            print(f"Warning: could not read best_sisdr.pt metrics: {exc}")

    running_loss = 0.0
    best_si_sdr = -float("inf")
    running_similarity = 0.0

    log_times = {"data": 0, "forward": 0, "backward": 0, "optim": 0}
    last_log_time = time.monotonic()

    for epoch in range(start_epoch + 1, epochs + 1):
        for i, batch in enumerate(loader, start=1):
            step += 1

            loss, sim, t = train_step(model, batch, optimizer, scaler, ema, device, use_cuda, log_level, mrstft_chunk_size, loss_type)
            running_loss += loss
            running_similarity += sim

            if log_level == 2:
                for key in log_times.keys():
                    log_times[key] += t[key]

            if i >= epoch_size:
                break

        if epoch % log_every == 0:
            last_log_time = log_status(epoch, running_loss, running_similarity, log_times, log_every, log_level, epoch_size, last_log_time)
            running_loss = 0.0
            running_similarity = 0.0
            log_times = {k: 0.0 for k in log_times}

        if epoch % eval_every == 0:
            metrics = run_eval(epoch, model, ema, device, data_root, data_format, sources, segment_seconds, max_batches=eval_batches, eval_num_workers=eval_num_workers)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            # Always keep latest metrics in periodic checkpoints; update best model only if SI-SDR improves.
            if metrics["si_sdr"] > best_si_sdr:
                best_si_sdr = metrics["si_sdr"]
                best_ckpt = ckpt_dir / "best_sisdr.pt"
                save_checkpoint(best_ckpt, model, optimizer, step, epoch, scaler,
                               extra={"ema": ema.state_dict(), "metrics": metrics})
                print(f"[Eval] New best model (SI-SDR={best_si_sdr:.2f} dB) has been saved to {best_ckpt}")

        if epoch % ckpt_every == 0:
            path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(path, model, optimizer, epoch, step, scaler,
                           extra={"ema": ema.state_dict(), "metrics": metrics if 'metrics' in locals() else {}})
            print(f"[CKPT] Saved checkpoint: {path}")

    # Final save
    ckpt_path = ckpt_dir / f"final_step_{step:06d}.pt"
    save_checkpoint(ckpt_path, model, optimizer, epoch, step, scaler,
                   extra={"ema": ema.state_dict(), "metrics": metrics if 'metrics' in locals() else {}})
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
    parser.add_argument("--workdir", type=str, default="./runs/hybrid_unet")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=400, help="Number of epochs to train")
    parser.add_argument("--epoch_size", type=int, default=50, help="Number of steps per epoch")
    parser.add_argument("--log_every", type=int, default=1, help="Log every N epochs")
    parser.add_argument("--ckpt_every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--segment_seconds", type=float, default=6.0, help="Segment length in seconds")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--items_per_epoch", type=int, default=10_000, help="Synthetic epoch size for MusdbRandomChunks")
    parser.add_argument("--eval_batches", type=int, default=20, help="Number of validation batches during eval")
    parser.add_argument("--eval_num_workers", type=int, default=0, help="DataLoader workers for evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help='path to checkpoint or "auto"')
    parser.add_argument("--log_level", type=int, default=0, choices=[0, 1, 2],
                        help="Logging level: 0 (minimal), 1 (+timestamp), 2 (detailed timings, worse performance)")
    parser.add_argument("--eval_every", type=int, default=5, help="Evaluate every N epochs")
    parser.add_argument("--use_checkpoint", action="store_true", help="Enable checkpointing in the model")
    parser.add_argument("--base_time", type=int, default=128, help="Base channel width for time-domain UNet")
    parser.add_argument("--base_freq", type=int, default=64, help="Base channel width for auxiliary branch")
    parser.add_argument("--mrstft_chunk_size", type=int, default=2, help="Number of (batch*source) waveforms per MR-STFT chunk")
    parser.add_argument("--model_type", type=str, default="hybrid", choices=["hybrid", "unet1d"], help="Model architecture to use")
    parser.add_argument("--loss_type", type=str, default="si_sdr_pit", choices=["si_sdr", "si_sdr_pit", "si_sdr_pit_mrstft", "legacy"], help="Loss configuration")

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
        log_level=args.log_level,
        eval_every=args.eval_every,
        items_per_epoch=args.items_per_epoch,
        eval_batches=args.eval_batches,
        use_checkpoint=args.use_checkpoint,
        eval_num_workers=args.eval_num_workers,
        base_time=args.base_time,
        base_freq=args.base_freq,
        mrstft_chunk_size=args.mrstft_chunk_size,
        model_type=args.model_type,
        loss_type=args.loss_type,
    )
