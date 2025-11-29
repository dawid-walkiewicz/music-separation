from typing import Callable, Iterable, Tuple

import torch
import torch.nn.functional as F
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
try:
    from auraloss.freq import MultiResolutionSTFTLoss
except ImportError:
    MultiResolutionSTFTLoss = None
    print("Warning: auraloss package not found. MRSTFT loss will not be available.")


def _flatten_bs(x: torch.Tensor) -> torch.Tensor:
    """Merge (B, S, C, L) or (B, S, L) into (BS, C, L) or (BS, L)."""
    if x.dim() == 4:
        b, s, c, l = x.shape
        return x.reshape(b * s, c, l)
    elif x.dim() == 3:
        b, s, l = x.shape
        return x.reshape(b * s, l)
    else:
        return x


def si_sdr_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Negative SI-SDR loss (higher SI-SDR -> lower loss).
    preds, targets: (B, S, 1, L) or (B, S, L)
    Returns scalar loss.
    """
    if preds.dim() == 4:
        preds = preds.squeeze(2)  # (B, S, L)
        targets = targets.squeeze(2)

    # Flatten batch and sources
    s_hat = _flatten_bs(preds)  # (BS, L)
    s = _flatten_bs(targets)

    si_sdr_val = scale_invariant_signal_noise_ratio(s_hat, s)

    loss = -torch.mean(si_sdr_val)
    return loss


def l1_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if preds.shape != targets.shape:
        # match length on time axis
        L = min(preds.shape[-1], targets.shape[-1])
        preds = preds[..., :L]
        targets = targets[..., :L]
    return F.l1_loss(preds, targets)


def mrstft_loss(preds: torch.Tensor, targets: torch.Tensor,
                ffts: Iterable[Tuple[int, int, int]] | None = None) -> torch.Tensor:
    """
    Multi-resolution STFT loss: sum of spectral convergence and log-magnitude L1 across resolutions.
    preds, targets: (B, S, 1, L)
    """
    if MultiResolutionSTFTLoss is None:
        raise ImportError("auraloss package is required for MRSTFT loss. Please install it via 'pip install auraloss'.")

    if ffts is None:
        ffts = (
            (512, 128, 512),
            (1024, 256, 1024),
            (2048, 512, 2048),
        )

    fft_sizes = [f[0] for f in ffts]
    hop_sizes = [f[1] for f in ffts]
    win_lengths = [f[2] for f in ffts]

    loss_mod = MultiResolutionSTFTLoss(
        fft_sizes=fft_sizes,
        hop_sizes=hop_sizes,
        win_lengths=win_lengths,
        w_sc=1.0,
        w_log_mag=1.0
    ).to(preds.device)

    if preds.dim() == 4:
        preds = preds.squeeze(2)  # (B, S, L)
        targets = targets.squeeze(2)

    x = _flatten_bs(preds)  # (BS, L)
    y = _flatten_bs(targets)  # (BS, L)

    x = x.unsqueeze(1)
    y = y.unsqueeze(1)

    loss = loss_mod(x, y)
    return loss


def get_loss_fn(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    name = name.lower()
    if name == 'l1':
        return l1_loss
    if name == 'si_sdr':
        return si_sdr_loss
    if name == 'mrstft':
        return mrstft_loss
    if name in ('si_sdr_l1', 'hybrid'):
        def _hybrid(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return 0.5 * si_sdr_loss(preds, targets) + 0.5 * l1_loss(preds, targets)
        return _hybrid
    raise ValueError(f"Unknown loss name: {name}. Choose from: l1, si_sdr, mrstft, si_sdr_l1, combo, perceptual")
