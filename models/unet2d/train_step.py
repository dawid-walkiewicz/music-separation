import torch
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from typing import Dict, Callable

from models.audio_functions import ensure_stereo
from models.unet2d.common_functions import ensure_target_wave_dims
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

    mixture_stereo, B, L = ensure_stereo(mixture)

    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
        loss_sum = None
        n_terms = 0

        for b in range(B):
            wav_stereo = mixture_stereo[b]  # (2, L)

            masked_stfts = model(wav_stereo)  # dict[name -> (2, F, T, 1) complex]

            for stem_name, pred_stft in masked_stfts.items():
                if stem_name not in targets_dict:
                    raise KeyError(
                        f"Stem '{stem_name}' returned by model.forward is not in dataset targets: {list(targets_dict.keys())}"
                    )

                target_wave = targets_dict[stem_name][b].to(device, non_blocking=True)  # (C, L)

                target_wave = ensure_target_wave_dims(target_wave, L)

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