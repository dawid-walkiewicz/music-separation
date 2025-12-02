import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Dict, Any

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


class SpectrogramFrontend(nn.Module):
    def __init__(self, n_fft: int = 2048, hop_length: int = 512, win_length: int | None = None, center: bool = True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.center = center
        window = torch.hann_window(self.win_length)
        self.register_buffer("window", window, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(x.device),
            center=self.center,
            return_complex=True,
        )


class Conv2DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FreqUNet2D(nn.Module):
    def __init__(self, in_channels: int = 1, base: int = 48, depth: int = 4, max_channels: int = 384, out_channels: int = 192,
                 use_checkpoint: bool = False):
        super().__init__()
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.enc_channels = []
        self.use_checkpoint = use_checkpoint and grad_checkpoint is not None

        ch = base
        in_ch = in_channels
        for idx in range(depth):
            block = nn.Sequential(Conv2DBlock(in_ch, ch), Conv2DBlock(ch, ch))
            self.enc_blocks.append(block)
            self.enc_channels.append(ch)
            if idx < depth - 1:
                self.downs.append(nn.AvgPool2d(kernel_size=2, stride=2))
                in_ch = ch
                ch = min(max_channels, ch * 2)

        self.up_blocks = nn.ModuleList()
        cur_ch = self.enc_channels[-1]
        for skip_ch in reversed(self.enc_channels[:-1]):
            block = nn.Sequential(
                Conv2DBlock(cur_ch + skip_ch, skip_ch),
                Conv2DBlock(skip_ch, skip_ch),
            )
            self.up_blocks.append(block)
            cur_ch = skip_ch

        self.head = nn.Conv2d(cur_ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        y = x
        for idx, block in enumerate(self.enc_blocks):
            y = self._maybe_checkpoint(block, y)
            skips.append(y)
            if idx < len(self.enc_blocks) - 1:
                y = self.downs[idx](y)

        for idx, block in enumerate(self.up_blocks):
            skip = skips[-(idx + 2)]
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = torch.cat([y, skip], dim=1)
            y = self._maybe_checkpoint(block, y)

        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(y)

    def _maybe_checkpoint(self, module: nn.Module, *args: torch.Tensor) -> torch.Tensor:
        tensors = [t for t in args if isinstance(t, torch.Tensor)]
        requires_grad = any(t.requires_grad for t in tensors)
        if self.use_checkpoint and requires_grad:
            def forward_fn(*inputs):
                return module(*inputs)

            return grad_checkpoint(forward_fn, *args, use_reentrant=True)
        return module(*args)


class TemporalTasNetPath(nn.Module):
    def __init__(self, in_ch: int = 1, hidden: int = 128, layers: int = 6, use_checkpoint: bool = False):
        super().__init__()
        self.pre = nn.Conv1d(in_ch, hidden, kernel_size=5, padding=2)
        self.blocks = nn.ModuleList(
            [ResidualTCNBlock(hidden, dilation=2 ** (i % 4), kernel_size=3) for i in range(layers)]
        )
        self.out_proj = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.use_checkpoint = use_checkpoint and grad_checkpoint is not None

    def forward(self, x: torch.Tensor, target_frames: int) -> torch.Tensor:
        y = self._maybe_checkpoint(self.pre, x)
        for block in self.blocks:
            y = self._maybe_checkpoint(block, y)
        y = self._maybe_checkpoint(self.out_proj, y)
        return F.interpolate(y, size=target_frames, mode="linear", align_corners=False)

    def _maybe_checkpoint(self, module: nn.Module, *args: torch.Tensor) -> torch.Tensor:
        tensors = [t for t in args if isinstance(t, torch.Tensor)]
        requires_grad = any(t.requires_grad for t in tensors)
        if self.use_checkpoint and requires_grad:
            def forward_fn(*inputs):
                return module(*inputs)

            return grad_checkpoint(forward_fn, *args, use_reentrant=True)
        return module(*args)


class HybridFusionHead(nn.Module):
    def __init__(self, freq_ch: int, time_ch: int, n_sources: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(freq_ch + time_ch, freq_ch, kernel_size=1),
            nn.BatchNorm2d(freq_ch),
            nn.GELU(),
            nn.Conv2d(freq_ch, n_sources, kernel_size=1),
        )

    def forward(self, freq_feat: torch.Tensor, time_feat_2d: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([freq_feat, time_feat_2d], dim=1)
        return torch.sigmoid(self.net(fused))


class HybridTimeFreqUNet(nn.Module):
    def __init__(
            self,
            n_sources: int,
            stft_fft: int,
            stft_hop: int,
            stft_win: int | None,
            freq_base: int = 48,
            freq_depth: int = 4,
            freq_out: int = 192,
            time_hidden: int = 128,
            time_layers: int = 8,
            use_checkpoint: bool = False,
    ):
        super().__init__()
        self.frontend = SpectrogramFrontend(n_fft=stft_fft, hop_length=stft_hop, win_length=stft_win)
        self.freq_unet = FreqUNet2D(in_channels=1, base=freq_base, depth=freq_depth, out_channels=freq_out,
                                    use_checkpoint=use_checkpoint)
        self.temporal_path = TemporalTasNetPath(in_ch=1, hidden=time_hidden, layers=time_layers,
                                                use_checkpoint=use_checkpoint)
        self.mask_head = HybridFusionHead(freq_out, time_hidden, n_sources)

    def forward(self, mixture: torch.Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            spec = self.frontend(mixture.float())
        magnitude = torch.log1p(spec.abs()).unsqueeze(1).to(mixture.dtype)
        freq_feat = self.freq_unet(magnitude)
        del magnitude

        time_feat = self.temporal_path(mixture, target_frames=freq_feat.shape[-1])  # (B, C, T)
        time_feat_2d = time_feat.unsqueeze(2).expand(-1, -1, freq_feat.shape[-2], -1)

        masks = self.mask_head(freq_feat, time_feat_2d)
        return {
            "mask": masks,
            "stft": {
                "spec": spec,
                "n_fft": self.frontend.n_fft,
                "hop_length": self.frontend.hop_length,
                "win_length": self.frontend.win_length,
                "window": self.frontend.window,
                "center": self.frontend.center,
            }
        }


class UNet1D(nn.Module):
    """
    Simple 1D U-Net for time-domain source separation.
    Input: mixture (C, L) where C=1, L=segment_samples
    Output: masks for S sources: (S, C, L). Multiplied by mixture -> predicted sources.
    """

    def __init__(
            self,
            n_sources: int = 4,
            base: int = 64,
            variant: str = "unet",
            use_checkpoint: bool = False,
            stft_fft: int = 2048,
            stft_hop: int = 512,
            stft_win_length: int | None = None,
            freq_base: int = 48,
            freq_depth: int = 4,
            freq_out: int = 192,
            time_hidden: int = 128,
            time_layers: int = 8,
    ):
        super().__init__()
        self.n_sources = n_sources
        self.variant = variant.lower()
        if self.variant not in {"unet", "hybrid"}:
            raise ValueError("variant must be 'unet' or 'hybrid'")

        if use_checkpoint and grad_checkpoint is None:
            warnings.warn("Gradient checkpointing requested but torch.utils.checkpoint is unavailable.")
        self.use_checkpoint = use_checkpoint and grad_checkpoint is not None
        self.hybrid_branch: HybridTimeFreqUNet | None = None
        if self.variant == "unet":
            self._build_unet(base)
        else:
            self.hybrid_branch = HybridTimeFreqUNet(
                n_sources=n_sources,
                stft_fft=stft_fft,
                stft_hop=stft_hop,
                stft_win=stft_win_length,
                freq_base=freq_base,
                freq_depth=freq_depth,
                freq_out=freq_out,
                time_hidden=time_hidden,
                time_layers=time_layers,
                use_checkpoint=self.use_checkpoint,
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor | Dict[str, Any]:
        if self.variant == "hybrid" and self.hybrid_branch is not None:
            return self.hybrid_branch(mixture)
        return self._forward_unet(mixture)

    def _build_unet(self, base: int) -> None:
        ch = base
        self.e1 = nn.Sequential(ConvBlock(1, ch), ConvBlock(ch, ch))
        self.d1 = nn.Conv1d(ch, ch, 4, 2, 1)

        self.e2 = nn.Sequential(ConvBlock(ch, ch * 2), ConvBlock(ch * 2, ch * 2))
        self.d2 = nn.Conv1d(ch * 2, ch * 2, 4, 2, 1)

        self.e3 = nn.Sequential(ConvBlock(ch * 2, ch * 4), ConvBlock(ch * 4, ch * 4))
        self.d3 = nn.Conv1d(ch * 4, ch * 4, 4, 2, 1)

        self.b = nn.Sequential(
            ConvBlock(ch * 4, ch * 8),
            nn.Dropout(0.2),
            ConvBlock(ch * 8, ch * 8),
        )

        self.u3 = nn.ConvTranspose1d(ch * 8, ch * 4, 4, 2, 1)
        self.p3 = nn.Sequential(ConvBlock(ch * 8, ch * 4), ConvBlock(ch * 4, ch * 4))

        self.u2 = nn.ConvTranspose1d(ch * 4, ch * 2, 4, 2, 1)
        self.p2 = nn.Sequential(ConvBlock(ch * 4, ch * 2), ConvBlock(ch * 2, ch * 2))

        self.u1 = nn.ConvTranspose1d(ch * 2, ch, 4, 2, 1)
        self.p1 = nn.Sequential(ConvBlock(ch * 2, ch), ConvBlock(ch, ch))

        self.out = nn.Conv1d(ch, self.n_sources, 1)
        self.act_out = nn.Softmax(dim=1)

    def _forward_unet(self, mixture: torch.Tensor) -> torch.Tensor:
        x1 = self._maybe_checkpoint(self.e1, mixture)
        x2_in = F.leaky_relu(self.d1(x1), 0.1)
        x2 = self._maybe_checkpoint(self.e2, x2_in)
        x3_in = F.leaky_relu(self.d2(x2), 0.1)
        x3 = self._maybe_checkpoint(self.e3, x3_in)
        xb_in = F.leaky_relu(self.d3(x3), 0.1)
        xb = self._maybe_checkpoint(self.b, xb_in)

        y3 = self.u3(xb)
        y3 = torch.cat([y3, x3], dim=1)
        y3 = self._maybe_checkpoint(self.p3, y3)

        y2 = self.u2(y3)
        y2 = torch.cat([y2, x2], dim=1)
        y2 = self._maybe_checkpoint(self.p2, y2)

        y1 = self.u1(y2)
        y1 = torch.cat([y1, x1], dim=1)
        y1 = self._maybe_checkpoint(self.p1, y1)

        masks = self.act_out(self.out(y1))
        return masks

    def _maybe_checkpoint(self, module: nn.Module, *args: torch.Tensor) -> torch.Tensor:
        tensors = [t for t in args if isinstance(t, torch.Tensor)]
        requires_grad = any(t.requires_grad for t in tensors)
        if self.use_checkpoint and self.training and requires_grad:
            def forward_fn(*inputs):
                return module(*inputs)

            return grad_checkpoint(forward_fn, *args, use_reentrant=True)
        return module(*args)


def apply_masks(mixture: torch.Tensor, masks: torch.Tensor | Dict[str, Any]) -> torch.Tensor:
    """Apply masks either in time domain (classic UNet) or STFT domain (hybrid)."""
    if isinstance(masks, dict):
        mask = masks["mask"]
        stft_info = masks["stft"]
        spec = stft_info["spec"]
        window = stft_info["window"].to(spec.device)
        masked_spec = spec.unsqueeze(1) * mask  # (B, S, F, T)
        B, S, Freq, Frames = masked_spec.shape
        masked_spec = masked_spec.reshape(B * S, Freq, Frames)
        sources = torch.istft(
            masked_spec,
            n_fft=stft_info["n_fft"],
            hop_length=stft_info["hop_length"],
            win_length=stft_info["win_length"],
            window=window,
            center=stft_info.get("center", True),
            length=mixture.shape[-1],
        )
        sources = sources.reshape(B, S, -1)
        return sources.unsqueeze(2)

    Lc = min(mixture.shape[-1], masks.shape[-1])
    mixture = mixture[..., :Lc]
    masks = masks[..., :Lc]
    return masks.unsqueeze(2) * mixture.unsqueeze(1)
