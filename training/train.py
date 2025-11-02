from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from training.data import MusdbRandomChunks
from training.model import UNet1D, apply_masks
from training.utils import load_checkpoint, save_checkpoint, find_latest_checkpoint

torch.backends.cudnn.benchmark = True


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


def train(
        data_root: str = "./musdb18-wav",
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
        sources=list(sources),
        segment_seconds=segment_seconds,
        items_per_epoch=10_000,
        mono=True,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None)
    )

    model = UNet1D(n_sources=len(sources), base=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
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
    last_log_time = time.monotonic()
    acc_data_time = 0.0
    acc_forward_time = 0.0
    acc_backward_time = 0.0
    acc_optimizer_time = 0.0

    step_start_time = time.monotonic()

    while step < max_steps:
        data_start_time = time.monotonic()

        for batch in loader:
            if use_cuda:
                torch.cuda.synchronize()
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
                targets_c = targets[..., :Lp]  # dopasuj długość
                loss = (preds - targets_c).abs().mean()

            if use_cuda:
                torch.cuda.synchronize()
            forward_end_time = time.monotonic()
            acc_forward_time += (forward_end_time - forward_start_time)

            backward_start_time = time.monotonic()

            scaler.scale(loss).backward()

            if use_cuda:
                torch.cuda.synchronize()
            backward_end_time = time.monotonic()
            acc_backward_time += (backward_end_time - backward_start_time)

            optimizer_start_time = time.monotonic()

            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            if use_cuda:
                torch.cuda.synchronize()
            optimizer_end_time = time.monotonic()
            acc_optimizer_time += (optimizer_end_time - optimizer_start_time)

            running_loss += float(loss.item())
            step += 1

            if step % log_every == 0:
                avg_loss = running_loss / log_every

                current_time = time.monotonic()
                total_duration_wall = current_time - last_log_time
                steps_per_sec = log_every / total_duration_wall

                avg_data_time_ms = (acc_data_time / log_every) * 1000
                avg_forward_time_ms = (acc_forward_time / log_every) * 1000
                avg_backward_time_ms = (acc_backward_time / log_every) * 1000
                avg_optimizer_time_ms = (acc_optimizer_time / log_every) * 1000

                total_measured_ms = avg_data_time_ms + avg_forward_time_ms + avg_backward_time_ms + avg_optimizer_time_ms

                print(f"\n--- Step {step:6d} ---")
                print(f"  Ogólnie:   {steps_per_sec:.2f} steps/s | Avg Loss: {avg_loss:.5f}")
                print(f"  Średnie czasy na krok (razem zmierzone: {total_measured_ms:.2f} ms):")

                if total_measured_ms > 0:
                    print(
                        f"    - Ładowanie Danych: {avg_data_time_ms:8.2f} ms ({avg_data_time_ms / total_measured_ms * 100:5.1f}%)")
                    print(
                        f"    - Forward Pass:     {avg_forward_time_ms:8.2f} ms ({avg_forward_time_ms / total_measured_ms * 100:5.1f}%)")
                    print(
                        f"    - Backward Pass:    {avg_backward_time_ms:8.2f} ms ({avg_backward_time_ms / total_measured_ms * 100:5.1f}%)")
                    print(
                        f"    - Optimizer/Update: {avg_optimizer_time_ms:8.2f} ms ({avg_optimizer_time_ms / total_measured_ms * 100:5.1f}%)")

                running_loss = 0.0
                last_log_time = current_time
                acc_data_time = 0.0
                acc_forward_time = 0.0
                acc_backward_time = 0.0
                acc_optimizer_time = 0.0

            if step % ckpt_every == 0:
                ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
                save_checkpoint(ckpt_path, model, optimizer, step, epoch, scaler)
                print(f"Zapisano checkpoint: {ckpt_path}")

            if step >= max_steps:
                break

            data_start_time = time.monotonic()
        epoch += 1

    # Zapis końcowy
    ckpt_path = ckpt_dir / f"final_step_{step:06d}.pt"
    save_checkpoint(ckpt_path, model, optimizer, step, epoch, scaler)
    print(f"Zapisano checkpoint końcowy: {ckpt_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trenowanie 1D U-Net do separacji źródeł audio (MUSDB18)")
    parser.add_argument("--data_root", type=str, default="./musdb18-wav")
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

    print(args)

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
