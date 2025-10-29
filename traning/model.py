from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _match_time(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Przytnij oba tensory do wspólnej (minimalnej) długości po osi czasu."""
    L = min(a.shape[-1], b.shape[-1])
    if a.shape[-1] != L:
        a = a[..., :L]
    if b.shape[-1] != L:
        b = b[..., :L]
    return a, b


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, k, s, p)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class UNet1D(nn.Module):
    """
    Prosty 1D U-Net do separacji źródeł w dziedzinie czasowej.
    Wejście: mixture (C, L) gdzie C=1, L=segment_samples
    Wyjście: maski dla S źródeł: (S, C, L). Mnożone przez mixture -> predykcja źródeł.
    """

    def __init__(self, n_sources: int = 4, base: int = 64):
        super().__init__()
        self.n_sources = n_sources
        ch = base

        # Encoder
        self.e1 = nn.Sequential(ConvBlock(1, ch), ConvBlock(ch, ch))
        self.d1 = nn.Conv1d(ch, ch, 4, 2, 1)

        self.e2 = nn.Sequential(ConvBlock(ch, ch * 2), ConvBlock(ch * 2, ch * 2))
        self.d2 = nn.Conv1d(ch * 2, ch * 2, 4, 2, 1)

        self.e3 = nn.Sequential(ConvBlock(ch * 2, ch * 4), ConvBlock(ch * 4, ch * 4))
        self.d3 = nn.Conv1d(ch * 4, ch * 4, 4, 2, 1)

        # Bottleneck
        self.b = nn.Sequential(ConvBlock(ch * 4, ch * 8), ConvBlock(ch * 8, ch * 8))

        # Decoder
        self.u3 = nn.ConvTranspose1d(ch * 8, ch * 4, 4, 2, 1)
        self.p3 = nn.Sequential(ConvBlock(ch * 8, ch * 4), ConvBlock(ch * 4, ch * 4))

        self.u2 = nn.ConvTranspose1d(ch * 4, ch * 2, 4, 2, 1)
        self.p2 = nn.Sequential(ConvBlock(ch * 4, ch * 2), ConvBlock(ch * 2, ch * 2))

        self.u1 = nn.ConvTranspose1d(ch * 2, ch, 4, 2, 1)
        self.p1 = nn.Sequential(ConvBlock(ch * 2, ch), ConvBlock(ch, ch))

        # Wyjście masek
        self.out = nn.Conv1d(ch, n_sources, 1)
        self.act_out = nn.Sigmoid()  # maski w [0,1]

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        # mixture: (B, C=1, L)
        x1 = self.e1(mixture)  # (B, ch, L)
        x2 = self.e2(F.leaky_relu(self.d1(x1), 0.1))  # (B, 2ch, L/2)
        x3 = self.e3(F.leaky_relu(self.d2(x2), 0.1))  # (B, 4ch, L/4)
        xb = self.b(F.leaky_relu(self.d3(x3), 0.1))   # (B, 8ch, L/8)

        y3 = self.u3(xb)  # (B, 4ch, ~L/4)
        y3, x3 = _match_time(y3, x3)
        y3 = torch.cat([y3, x3], dim=1)
        y3 = self.p3(y3)

        y2 = self.u2(y3)  # (B, 2ch, ~L/2)
        y2, x2 = _match_time(y2, x2)
        y2 = torch.cat([y2, x2], dim=1)
        y2 = self.p2(y2)

        y1 = self.u1(y2)  # (B, ch, ~L)
        y1, x1 = _match_time(y1, x1)
        y1 = torch.cat([y1, x1], dim=1)
        y1 = self.p1(y1)

        masks = self.act_out(self.out(y1))  # (B, S, ~L)
        return masks


def apply_masks(mixture: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    mixture: (B, 1, L)
    masks: (B, S, Lm)
    -> sources: (B, S, 1, Lc) gdzie Lc = min(L, Lm)
    """
    Lc = min(mixture.shape[-1], masks.shape[-1])
    mixture = mixture[..., :Lc]
    masks = masks[..., :Lc]
    return masks.unsqueeze(2) * mixture.unsqueeze(1)
