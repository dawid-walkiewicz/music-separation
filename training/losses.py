from typing import Iterable, Tuple, Optional
from collections.abc import Callable

import torch
import torch.nn.functional as F
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


def _ensure_pair_shape(preds: torch.Tensor, targets: torch.Tensor):
    """Ensure preds and targets have compatible shapes. Return (preds_flat, targets_flat).

    Accepts inputs of shape (B, S, 1, L), (B, S, L) or (BS, L) and returns tensors
    of shape (BS, L).
    """
    # Squeeze channel dim if present
    if preds.dim() == 4:
        preds = preds.squeeze(2)
    if targets.dim() == 4:
        targets = targets.squeeze(2)

    # If inputs are (B, S, L) -> (BS, L)
    preds_flat = _flatten_bs(preds)
    targets_flat = _flatten_bs(targets)

    # Make sure time lengths match
    if preds_flat.shape[-1] != targets_flat.shape[-1]:
        L = min(preds_flat.shape[-1], targets_flat.shape[-1])
        preds_flat = preds_flat[..., :L]
        targets_flat = targets_flat[..., :L]

    return preds_flat, targets_flat


def si_sdr(
    preds: torch.Tensor,
    targets: torch.Tensor,
    zero_mean: bool = True,
    eps: float = 1e-8,
    reduction: Optional[str] = "none",
) -> torch.Tensor:
    """Compute SI-SDR (dB).

    Args:
        preds, targets: (B, S, 1, L) or (B, S, L) or (BS, L).
        zero_mean: whether to remove mean per-sample (recommended for SI-SDR).
        reduction: 'none' to return per-sample tensor of shape (BS,), 'mean' to return scalar.

    Returns:
        Tensor of SI-SDR values in dB (per-sample or mean).
    """
    s_hat, s = _ensure_pair_shape(preds, targets)

    if zero_mean:
        s_hat = s_hat - s_hat.mean(dim=-1, keepdim=True)
        s = s - s.mean(dim=-1, keepdim=True)

    # dot = <s_hat, s>
    dot = (s_hat * s).sum(dim=-1)
    s_energy = (s ** 2).sum(dim=-1) + eps
    proj = (dot / s_energy).unsqueeze(-1) * s
    e_noise = s_hat - proj

    si_num = (proj ** 2).sum(dim=-1)
    si_den = (e_noise ** 2).sum(dim=-1) + eps
    si_sdr_val = 10.0 * torch.log10(si_num / si_den + eps)

    if reduction == "mean":
        return si_sdr_val.mean()
    return si_sdr_val


def sdr(
    preds: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-8,
    reduction: Optional[str] = "none",
) -> torch.Tensor:
    """Compute simple SDR (dB): 10*log10(||s||^2 / ||s - s_hat||^2).

    Accepts same shapes as `si_sdr` and returns per-sample or mean SDR.
    """
    s_hat, s = _ensure_pair_shape(preds, targets)

    num = (s ** 2).sum(dim=-1)
    den = ((s - s_hat) ** 2).sum(dim=-1) + eps
    sdr_val = 10.0 * torch.log10(num / den + eps)

    if reduction == "mean":
        return sdr_val.mean()
    return sdr_val


def si_sdr_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Negative SI-SDR loss (lower is better). Returns scalar.
    preds, targets: (B, S, 1, L) or (B, S, L)
    """
    # we want a scalar loss to minimize -> negative mean SI-SDR
    val = si_sdr(preds, targets, reduction="none")
    return -torch.mean(val)


def l1_loss(preds: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """L1 loss that handles mismatched time lengths and common shapes.

    Args:
        reduction: 'mean' (default) or 'sum' or 'none'.
    """
    L = min(preds.shape[-1], targets.shape[-1])
    preds = preds[..., :L]
    targets = targets[..., :L]

    if reduction == "mean":
        return F.l1_loss(preds, targets)
    elif reduction == "sum":
        return F.l1_loss(preds, targets, reduction="sum")
    else:
        return F.l1_loss(preds, targets, reduction="none")


def l2_loss(preds: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """L2 loss that handles mismatched time lengths and common shapes.

    Args:
        reduction: 'mean' (default) or 'sum' or 'none'.
    """
    L = min(preds.shape[-1], targets.shape[-1])
    preds = preds[..., :L]
    targets = targets[..., :L]

    if reduction == "mean":
        return F.mse_loss(preds, targets)
    elif reduction == "sum":
        return F.mse_loss(preds, targets, reduction="sum")
    else:
        return F.mse_loss(preds, targets, reduction="none")


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
    """Return a loss function by name. The returned callable accepts (preds, targets) and
    returns a scalar tensor suitable for training.
    """
    name = name.lower()
    if name == 'l1':
        return lambda p, t: l1_loss(p, t, reduction="mean")
    if name == 'l2':
        return lambda p, t: l2_loss(p, t, reduction="mean")
    if name == 'si_sdr':
        return si_sdr_loss
    if name == 'mrstft':
        return mrstft_loss
    if name in ('si_sdr_l1', 'hybrid'):
        def _hybrid(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return 0.5 * si_sdr_loss(preds, targets) + 0.5 * l1_loss(preds, targets, reduction="mean")
        return _hybrid
    raise ValueError(f"Unknown loss name: {name}. Choose from: l1, si_sdr, mrstft, si_sdr_l1, hybrid")
