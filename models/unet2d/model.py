import math
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.unet2d.unet import UNet


def pad_to_length(tensor: Tensor, target_length: int) -> Tensor:
    """
    Pads or cuts the input tensor to match the target length.
    Args:
        tensor (Tensor): Input tensor of shape (..., L).
        target_length (int): Desired length .
    Returns:
        Tensor: Tensor of shape (..., target_length).
    """
    orig_size = tensor.size(-1)
    if orig_size >= target_length:
        return tensor[..., :target_length]
    else:
        pad_size = target_length - orig_size
        return F.pad(tensor, [0, pad_size])


class Unet2DWrapper(nn.Module):
    def __init__(self, stem_names: List[str] = None, layers: int = 6):
        super().__init__()

        assert stem_names, "Must provide stem names."
        # stft config
        self.F = 2048
        self.T = 512  # ~11.6s of audio
        self.win_length = 4096
        self.hop_length = 1024
        self.win = nn.Parameter(torch.hann_window(self.win_length), requires_grad=False)

        self.stem_nets = nn.ModuleDict({name: UNet(n_layers=layers) for name in stem_names})

    def compute_stft(self, wav: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Computes STFT features from waveform.

        Args:
            wav (Tensor): shape (2, L) or (B, L). We use stereo (2, L).

        Returns:
            stft (Tensor): complex tensor of shape (2, F, L) truncated to self.F frequencies.
            mag (Tensor):  magnitude tensor of shape (2, F, L).
        """
        stft_c = torch.stft(
            wav,
            n_fft=self.win_length,
            hop_length=self.hop_length,
            window=self.win,
            center=True,
            return_complex=True,
            pad_mode="constant",
        )  # (C, F_tot, L)

        # only keep freqs smaller than self.F
        stft_c = stft_c[:, : self.F, :]  # (C, F, L)
        mag = stft_c.abs()  # (C, F, L)

        return stft_c, mag

    def inverse_stft(self, stft: Tensor) -> Tensor:
        """Inverse STFT back to waveform.

        Args:
            stft: complex tensor with shape (C, F, L).

        Returns:
            wav: Tensor of shape (C, L) (stereo for Splitter).
        """
        # Pad frequency axis back to win_length//2+1 as expected by istft
        # Current stft_c: (C, F, L), F <= self.win_length//2+1
        pad = self.win_length // 2 + 1 - stft.size(1)
        stft = F.pad(stft, (0, 0, 0, max(0, pad)))

        wav = torch.istft(
            stft,
            n_fft=self.win_length,
            hop_length=self.hop_length,
            center=True,
            window=self.win,
        )  # (C, L)
        return wav

    def forward(self, wav: Tensor) -> Dict[str, Tensor]:
        """
        Separates stereo wav into different tracks (1 predicted track per stem)
        Args:
            wav (tensor): 2 x L

        Returns:
            specs (Dict[name, Tensor]): masked stfts by track name (2 x F x L)
        """
        stft, stft_mag = self.compute_stft(wav)

        L = stft_mag.size(2)

        stft_mag_feat = stft_mag.unsqueeze(0)  # (1, 2, F, L)
        stft_mag_feat = pad_to_length(stft_mag_feat, self.T)  # (1, 2, F, T)
        stft_mag_feat = stft_mag_feat.transpose(2, 3)  # (1, 2, T, F)

        # compute stems' masks in feature space
        masks = {name: net(stft_mag_feat) for name, net in self.stem_nets.items()}  # each: (B, 2, T, F)

        mask_sum = sum([m ** 2 for m in masks.values()]) # B x 2 x T x F
        mask_sum += 1e-10  # avoid div by zero

        def apply_mask(mask: Tensor) -> Tensor:
            # normalize masks
            mask = (mask ** 2) / mask_sum
            mask = mask.transpose(2, 3)  # B x 2 x F x T

            # match original STFT length
            mask = mask.squeeze(0)[:, :, :L]  # 2 x F x L (real mask)
            stft_masked = stft * mask
            return stft_masked

        return {name: apply_mask(m) for name, m in masks.items()}

    def separate(self, wav: Tensor) -> Dict[str, Tensor]:
        """
        Separates stereo wav into different tracks (1 predicted track per stem)
        Args:
            wav (tensor): 2 x L
        Returns:
            wavs by track name
        """

        stft_masks = self.forward(wav)

        return {
            name: self.inverse_stft(stft_masked)
            for name, stft_masked in stft_masks.items()
        }
