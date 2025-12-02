"""Quick sanity test for HybridTimeFreqUNet.

Runs a couple of synthetic training steps to verify that the forward/backward
pass works and the loss decreases, signalling the network can learn.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import statistics

import torch

# Ensure project root is on sys.path for `training` package imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.model import (
    HybridTimeFreqUNet,
    reconstruct_sources,
    pit_si_sdr_loss,
    mrstft_loss,
)


def run_sanity_test(
    device: torch.device,
    batch_size: int = 2,
    seq_len: int = 32768,
    n_sources: int = 4,
    steps: int = 3,
    base_time: int = 128,
    base_freq: int = 64,
) -> dict:
    torch.manual_seed(0)
    use_cuda = device.type == "cuda"
    model = HybridTimeFreqUNet(
        n_sources=n_sources,
        base_time=base_time,
        base_freq=base_freq,
        n_fft=1024,
        hop_length=256,
        use_checkpoint=False,
    ).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    losses: list[float] = []

    for step in range(steps):
        mixture = torch.randn(batch_size, 1, seq_len, device=device)
        # Create ground-truth masks that sum to one, so targets are consistent with mixture
        true_masks = torch.rand(batch_size, n_sources, seq_len, device=device)
        true_masks = true_masks / true_masks.sum(dim=1, keepdim=True)
        targets = true_masks * mixture

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_cuda):
            logits = model(mixture)
            preds = reconstruct_sources(mixture, logits)
            preds, targets = preds[..., :seq_len], targets[..., :preds.shape[-1]]
            pit_loss = pit_si_sdr_loss(preds, targets)
            stft_loss = mrstft_loss(
                preds,
                targets,
                fft_sizes=(256, 512),
                hop_factors=(0.5, 0.25),
                chunk_size=batch_size,
            )
            loss = pit_loss + 0.1 * stft_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(float(loss.item()))
        print(f"Step {step+1}/{steps} | loss={losses[-1]:.4f}")

    loss_drop = losses[0] - losses[-1] if len(losses) > 1 else 0.0
    stats = {
        "losses": losses,
        "mean_loss": statistics.fmean(losses),
        "loss_drop": loss_drop,
    }
    print(f"Loss drop over {steps} steps: {loss_drop:.4f}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Quick sanity test for HybridTimeFreqUNet")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--steps", type=int, default=3, help="Number of synthetic training steps")
    parser.add_argument("--seq_len", type=int, default=32768, help="Sequence length in samples")
    parser.add_argument("--batch_size", type=int, default=2, help="Synthetic batch size")
    parser.add_argument("--base_time", type=int, default=128, help="Base channels for time UNet")
    parser.add_argument("--base_freq", type=int, default=64, help="Base channels for freq branch")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats = run_sanity_test(
        device=device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        steps=args.steps,
        base_time=args.base_time,
        base_freq=args.base_freq,
    )

    print("\nSummary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
