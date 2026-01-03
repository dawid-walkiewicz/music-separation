from torch import Tensor

def ensure_target_wave_dims(
        target_wave: Tensor,
        L: int,
)-> Tensor:
    """
    Ensure that the target waveform has the correct dimensions.

    Args:
        target_wave (torch.Tensor): Target audio tensor of shape (C, L)
        L (int): Expected length of the audio signal.

    Returns:
        target_wave (torch.Tensor): Target audio tensor with ensured dimensions (2, L)
    """
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

    return target_wave