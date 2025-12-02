import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

try:
    from torch.utils.checkpoint import checkpoint as grad_checkpoint
except ImportError:  # pragma: no cover - checkpoint is available in recent PyTorch
    grad_checkpoint = None


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, k, s, p)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, groups=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class HybridTemporalPath(nn.Module):
    """Dual-path temporal module inspired by recent hybrid separators (TCN + U-Net)."""

    def __init__(self, in_ch: int, hidden: int, depth: int = 4):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, hidden, 1)
        dilations = [2 ** i for i in range(depth)]
        self.blocks = nn.ModuleList([ResidualTCNBlock(hidden, d) for d in dilations])
        self.norm = nn.BatchNorm1d(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(x)
        for block in self.blocks:
            y = block(y)
        return self.norm(y)


class UNet1D(nn.Module):
    """
    Simple 1D U-Net for time-domain source separation.
    Input: mixture (C, L) where C=1, L=segment_samples
    Output: masks for S sources: (S, C, L). Multiplied by mixture -> predicted sources.
    """

    def __init__(self, n_sources: int = 4, base: int = 64, variant: str = "unet", use_checkpoint: bool = False):
        super().__init__()
        self.n_sources = n_sources
        self.variant = variant.lower()
        if self.variant not in {"unet", "hybrid"}:
            raise ValueError("variant must be 'unet' or 'hybrid'")

        if use_checkpoint and grad_checkpoint is None:
            warnings.warn("Gradient checkpointing requested but torch.utils.checkpoint is unavailable.")
        self.use_checkpoint = use_checkpoint and grad_checkpoint is not None
        ch = base

        # Encoder
        self.e1 = nn.Sequential(ConvBlock(1, ch), ConvBlock(ch, ch))
        self.d1 = nn.Conv1d(ch, ch, 4, 2, 1)

        self.e2 = nn.Sequential(ConvBlock(ch, ch * 2), ConvBlock(ch * 2, ch * 2))
        self.d2 = nn.Conv1d(ch * 2, ch * 2, 4, 2, 1)

        self.e3 = nn.Sequential(ConvBlock(ch * 2, ch * 4), ConvBlock(ch * 4, ch * 4))
        self.d3 = nn.Conv1d(ch * 4, ch * 4, 4, 2, 1)

        # Bottleneck
        self.b = nn.Sequential(
            ConvBlock(ch * 4, ch * 8),
            nn.Dropout(0.2),
            ConvBlock(ch * 8, ch * 8),
        )

        # Decoder
        self.u3 = nn.ConvTranspose1d(ch * 8, ch * 4, 4, 2, 1)
        self.p3 = nn.Sequential(ConvBlock(ch * 8, ch * 4), ConvBlock(ch * 4, ch * 4))

        self.u2 = nn.ConvTranspose1d(ch * 4, ch * 2, 4, 2, 1)
        self.p2 = nn.Sequential(ConvBlock(ch * 4, ch * 2), ConvBlock(ch * 2, ch * 2))

        self.u1 = nn.ConvTranspose1d(ch * 2, ch, 4, 2, 1)
        self.p1 = nn.Sequential(ConvBlock(ch * 2, ch), ConvBlock(ch, ch))

        # Output masks
        if self.variant == "hybrid":
            self.hybrid_path = HybridTemporalPath(1, ch)
            self.hybrid_fuse = nn.Sequential(
                nn.Conv1d(ch * 2, ch, 1),
                nn.BatchNorm1d(ch),
                nn.GELU(),
            )

        self.out = nn.Conv1d(ch, n_sources, 1)
        self.act_out = nn.Softmax(dim=1)  # masks in [0,1]

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        # mixture: (B, C=1, L)
        x1 = self._maybe_checkpoint(self.e1, mixture)  # (B, ch, L)
        x2_in = F.leaky_relu(self.d1(x1), 0.1)
        x2 = self._maybe_checkpoint(self.e2, x2_in)  # (B, 2ch, L/2)
        x3_in = F.leaky_relu(self.d2(x2), 0.1)
        x3 = self._maybe_checkpoint(self.e3, x3_in)  # (B, 4ch, L/4)
        xb_in = F.leaky_relu(self.d3(x3), 0.1)
        xb = self._maybe_checkpoint(self.b, xb_in)   # (B, 8ch, L/8)

        y3 = self.u3(xb)  # (B, 4ch, ~L/4)

        y3 = torch.cat([y3, x3], dim=1)
        y3 = self._maybe_checkpoint(self.p3, y3)

        y2 = self.u2(y3)  # (B, 2ch, ~L/2)
        y2 = torch.cat([y2, x2], dim=1)
        y2 = self._maybe_checkpoint(self.p2, y2)

        y1 = self.u1(y2)  # (B, ch, ~L)
        y1 = torch.cat([y1, x1], dim=1)
        y1 = self._maybe_checkpoint(self.p1, y1)

        if self.variant == "hybrid":
            hybrid_feat = self._maybe_checkpoint(self.hybrid_path, mixture)
            if hybrid_feat.shape[-1] != y1.shape[-1]:
                L = min(hybrid_feat.shape[-1], y1.shape[-1])
                hybrid_feat = hybrid_feat[..., :L]
                y1 = y1[..., :L]
            fused = torch.cat([y1, hybrid_feat], dim=1)
            y1 = self._maybe_checkpoint(self.hybrid_fuse, fused)

        masks = self.act_out(self.out(y1))  # (B, S, ~L)
        return masks

    def _maybe_checkpoint(self, module: nn.Module, *args: torch.Tensor) -> torch.Tensor:
        tensors = [t for t in args if isinstance(t, torch.Tensor)]
        requires_grad = any(t.requires_grad for t in tensors)
        if self.use_checkpoint and self.training and requires_grad:
            def forward_fn(*inputs):
                return module(*inputs)

            return grad_checkpoint(forward_fn, *args, use_reentrant=True)
        return module(*args)


def apply_masks(mixture: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    mixture: (B, 1, L)
    masks: (B, S, Lm)
    -> sources: (B, S, 1, Lc) where Lc = min(L, Lm)
    """
    Lc = min(mixture.shape[-1], masks.shape[-1])
    mixture = mixture[..., :Lc]
    masks = masks[..., :Lc]
    return masks.unsqueeze(2) * mixture.unsqueeze(1)
