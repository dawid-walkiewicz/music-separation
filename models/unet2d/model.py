import math
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.unet2d.unet import UNet


def batchify(tensor: Tensor, T: int) -> Tensor:
    """
    partition tensor into segments of length T, zero pad any ragged samples
    Args:
        tensor(Tensor): BxCxFxL
    Returns:
        tensor of size (B*[L/T] x C x F x T)
    """
    # Zero pad the original tensor to an even multiple of T
    orig_size = tensor.size(-1)
    new_size = math.ceil(orig_size / T) * T
    tensor = F.pad(tensor, [0, new_size - orig_size])
    # Partition the tensor into multiple samples of length T and stack them into a batch
    return torch.cat(torch.split(tensor, T, dim=-1), dim=0)


class Unet2DWrapper(nn.Module):
    def __init__(self, stem_names: List[str] = None):
        super(Unet2DWrapper, self).__init__()

        assert stem_names, "Must provide stem names."
        # stft config
        self.F = 2048
        self.T = 512
        self.win_length = 4096
        self.hop_length = 1024
        self.win = nn.Parameter(torch.hann_window(self.win_length), requires_grad=False)

        self.stems = nn.ModuleDict({name: UNet(in_channels=2) for name in stem_names})

    def compute_stft(self, wav: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes STFT features from waveform.

        Args:
            wav (Tensor): shape (2, L) or (B, L). For Splitter we use stereo (2, L).

        Returns:
            stft: complex tensor of shape (2, F, L, 1) truncated to self.F frequencies.
            mag:  magnitude tensor of shape (2, F, L).
        """
        # Ensure shape is (2, L)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)  # (1, L)
        # torch.stft with return_complex=True (recommended, future-proof)
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
        mag = stft_c.abs()               # (C, F, L)

        # For compatibility with existing code expecting last dim=2, we keep a 4D "stft" with trailing singleton dim
        stft = stft_c.unsqueeze(-1)      # (C, F, L, 1) complex view

        return stft, mag

    def inverse_stft(self, stft: Tensor) -> Tensor:
        """Inverse STFT back to waveform.

        Args:
            stft: complex tensor with shape (C, F, L, 1) or (C, F, L).

        Returns:
            wav: Tensor of shape (C, L) (stereo for Splitter).
        """
        # Collapse trailing singleton dim if present
        if stft.dim() == 4 and stft.size(-1) == 1:
            stft_c = stft.squeeze(-1)  # (C, F, L) complex
        else:
            stft_c = stft  # assume already (C, F, L) complex

        # Pad frequency axis back to win_length//2+1 as expected by istft
        # Current stft_c: (C, F, L), F <= self.win_length//2+1
        C, F_cur, L = stft_c.shape
        F_expected = self.win_length // 2 + 1
        if F_cur < F_expected:
            pad = F_expected - F_cur
            stft_c = F.pad(stft_c, (0, 0, 0, pad))  # pad frequency dimension at the end

        wav = torch.istft(
            stft_c,
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
            masked stfts by track name
        """
        # stft_c: 2 x F x L x 1 (complex), stft_mag: 2 x F x L (real)
        stft, stft_mag = self.compute_stft(wav.squeeze())

        L = stft_mag.size(2)

        # Prepare magnitude for UNet: 1 x 2 x F x T -> batchify -> B x 2 x T x F
        stft_mag_feat = stft_mag.unsqueeze(-1).permute(3, 0, 1, 2)  # (1, 2, F, L)
        stft_mag_feat = batchify(stft_mag_feat, self.T)             # (B, 2, F, T)
        stft_mag_feat = stft_mag_feat.transpose(2, 3)               # (B, 2, T, F)

        # compute stems' masks in feature space
        masks = {name: net(stft_mag_feat) for name, net in self.stems.items()}  # each: (B, 2, T, F)

        # compute denominator
        mask_sum = sum([m**2 for m in masks.values()])
        mask_sum += 1e-10

        def apply_mask(mask: Tensor) -> Tensor:
            # normalize masks
            mask = (mask**2 + 1e-10 / 2) / (mask_sum)
            mask = mask.transpose(2, 3)  # B x 2 x F x T

            # undo batchify: concat segments along time
            mask = torch.cat(torch.split(mask, 1, dim=0), dim=3)  # 1 x 2 x F x L

            # match original STFT length
            mask = mask.squeeze(0)[:, :, :L].unsqueeze(-1)  # 2 x F x L x 1 (real mask)
            # stft: 2 x F x L x 1 (complex) -> broadcast multiply
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