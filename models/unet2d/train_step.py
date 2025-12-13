import torch
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from typing import Dict, Any, Callable

from models.unet2d.model import Unet2DWrapper


def train_step(
    model: Unet2DWrapper,
    batch: Dict[str, object],
    optimizer: Optimizer,
    scaler: GradScaler,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    device: torch.device,
    use_cuda: bool,
) -> float:
    mixture = batch["mixture"].to(device, non_blocking=True)  # (B, C, L)
    targets_dict = batch["targets"]  # Dict[str, Tensor] with shape (B, C, L) per stem

    if mixture.dim() != 3:
        raise ValueError(f"Expected mixture of shape (B, C, L), got {mixture.shape}")
    B, C, L = mixture.shape
    if C == 1:
        mixture_stereo = mixture.repeat(1, 2, 1)  # (B, 2, L)
    elif C == 2:
        mixture_stereo = mixture
    else:
        raise ValueError(f"Splitter expects 1 or 2 channels, got C={C}")

    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
        loss_sum = None
        n_terms = 0

        for b in range(B):
            wav_stereo = mixture_stereo[b]  # (2, L)

            masked_stfts = model.forward(wav_stereo)  # dict[name -> (2, F, T, 1) complex]

            for stem_name, pred_stft in masked_stfts.items():
                if stem_name not in targets_dict:
                    raise KeyError(
                        f"Stem '{stem_name}' returned by model.forward is not in dataset targets: {list(targets_dict.keys())}"
                    )

                target_wave = targets_dict[stem_name][b].to(device, non_blocking=True)  # (C, L)

                if target_wave.dim() != 2:
                    raise ValueError(f"Expected target of shape (C, L), got {target_wave.shape}")

                tC, tL = target_wave.shape
                if tL != L:
                    raise ValueError(
                        f"Target and mixture length mismatch: target L={tL}, mix L={L}"
                    )

                if tC == 1:
                    target_wave = target_wave.repeat(2, 1)
                elif tC != 2:
                    raise ValueError(f"Expected target with 1 or 2 channels, got C={tC}")

                _, target_mag = model.compute_stft(target_wave)  # (2, F, T) real

                pred_mag = pred_stft.abs().squeeze(-1)

                loss = loss_fn(pred_mag, target_mag)

                if loss_sum is None:
                    loss_sum = loss
                else:
                    loss_sum = loss_sum + loss
                n_terms += 1

        if loss_sum is None or n_terms == 0:
            raise RuntimeError("No loss computed in train_step (empty batch or no stems)")

        total_loss: Tensor = loss_sum / n_terms

    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return float(total_loss.item())