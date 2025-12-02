import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import itertools
import warnings
warnings.filterwarnings(
    "ignore",
    message="None of the inputs have requires_grad=True"
)


def _conv_out_length(length: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1) -> int:
    numerator = length + 2 * padding - dilation * (kernel_size - 1) - 1
    return (numerator // stride) + 1


def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, k, s, p)
        self.norm = _group_norm(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class UNet1D(nn.Module):
    """Simple 1D U-Net for time-domain source separation.

    Input: mixture (C, L) where C=1, L=segment_samples
    Output: masks for S sources: (S, C, L). Multiplied by mixture -> predicted sources.
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
        self.b = nn.Sequential(
            ConvBlock(ch * 4, ch * 8),
            nn.Dropout(0.2),
            ConvBlock(ch * 8, ch * 8),
        )

        # Decoder
        self.u3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(ch * 8, ch * 4, kernel_size=3, padding=1),
        )
        self.p3 = nn.Sequential(ConvBlock(ch * 8, ch * 4), ConvBlock(ch * 4, ch * 4))

        self.u2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(ch * 4, ch * 2, kernel_size=3, padding=1),
        )
        self.p2 = nn.Sequential(ConvBlock(ch * 4, ch * 2), ConvBlock(ch * 2, ch * 2))

        self.u1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(ch * 2, ch, kernel_size=3, padding=1),
        )
        self.p1 = nn.Sequential(ConvBlock(ch * 2, ch), ConvBlock(ch, ch))

        # Output masks
        self.out = nn.Conv1d(ch, n_sources, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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

        logits = self.out(y1)  # (B, S, ~L)
        return logits


def _match_time(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop tensors along the time axis so they share the same length."""
    L = min(a.shape[-1], b.shape[-1])
    return a[..., :L], b[..., :L]


def apply_masks(mixture: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Apply masks to mixture to obtain separated sources.

    mixture: (B, 1, L)
    masks:   (B, S, Lm)
    -> sources: (B, S, 1, Lc) where Lc = min(L, Lm)
    """
    mixture, masks = _match_time(mixture, masks)
    return masks.unsqueeze(2) * mixture.unsqueeze(1)


class FreqBranch(nn.Module):
    """Temporal auxiliary branch operating on 1D waveforms.

    Takes mono input (B, 1, L) and produces compressed features (B, C, Lt).
    """

    def __init__(self, in_ch: int = 1, base: int = 32, out_ch: int | None = None, freq_bins: int | None = None):
        super().__init__()
        if out_ch is None:
            out_ch = base
        if freq_bins is None:
            raise ValueError("freq_bins must be provided for FreqBranch")
        self.freq_bins = freq_bins
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, kernel_size=3, padding=1),
            _group_norm(base),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base * 2, kernel_size=3, stride=2, padding=1),
            _group_norm(base * 2),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, kernel_size=3, stride=2, padding=1),
            _group_norm(base * 4),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(base * 4, out_ch, kernel_size=1)
        reduced_bins = _conv_out_length(
            _conv_out_length(freq_bins, kernel_size=3, stride=2, padding=1),
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.post_flatten = nn.Conv1d(out_ch * reduced_bins, out_ch, kernel_size=1)

    def forward(self, spec_mag: torch.Tensor) -> torch.Tensor:
        x = self.enc1(spec_mag)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.proj(x)
        B, C, Fp, T = x.shape
        x = x.reshape(B, C * Fp, T)
        return self.post_flatten(x)


class HybridTimeFreqUNet(nn.Module):
    """Hybrid model combining time-domain UNet1D with an auxiliary temporal branch.

    - Input:  mixture (B, 1, L)
    - Output: masks (B, S, Lt)

    Steps:
    1. UNet1D generates time-domain masks.
    2. FreqBranch (now temporal) processes the waveform for complementary cues.
    3. Features are aligned in time and concatenated.
    4. A Conv1d layer produces the final masks.
    """

    def __init__(
        self,
        n_sources: int = 4,
        base_time: int = 128,
        base_freq: int = 64,
        n_fft: int = 2048,
        hop_length: int = 512,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.n_sources = n_sources
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.use_checkpoint = use_checkpoint and grad_checkpoint is not None

        self.time_net = UNet1D(n_sources=n_sources, base=base_time)
        self.freq_branch = FreqBranch(
            in_ch=1,
            base=base_freq,
            out_ch=base_freq,
            freq_bins=(self.n_fft // 2) + 1,
        )
        self.freq_norm = nn.LayerNorm(base_freq)
        self.fuse_norm = nn.LayerNorm(n_sources + base_freq)
        self.fuse_conv = nn.Conv1d(n_sources + base_freq, n_sources, kernel_size=1)
        self.register_buffer("stft_window", torch.hann_window(self.n_fft))

    def _maybe_checkpoint(self, module: nn.Module, *args):
        if self.use_checkpoint and self.training:
            def forward_fn(*inputs):
                return module(*inputs)

            return grad_checkpoint(forward_fn, *args, use_reentrant=True)
        return module(*args)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        B, C, L = mixture.shape
        masks_time_logits = self._maybe_checkpoint(self.time_net, mixture)

        wav = mixture.squeeze(1)
        window = self.stft_window.to(wav.device)
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )
        mag = spec.abs().unsqueeze(1)
        mag = torch.log1p(mag + 1e-7)
        assert not torch.isnan(mag).any(), "NaNs detected after log compression"

        freq_feats = self._maybe_checkpoint(self.freq_branch, mag)
        freq_feats = self.freq_norm(freq_feats.transpose(1, 2)).transpose(1, 2)

        Lt = masks_time_logits.shape[-1]
        freq_feats_up = F.interpolate(freq_feats, size=Lt, mode="nearest")

        x = torch.cat([masks_time_logits, freq_feats_up], dim=1)
        x = self.fuse_norm(x.transpose(1, 2)).transpose(1, 2)
        logits = self.fuse_conv(x) + masks_time_logits
        return logits


def reconstruct_sources(mixture: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    mask_probs = torch.sigmoid(masks)
    preds = apply_masks(mixture, mask_probs)
    if preds.dim() == 4:
        preds = preds.squeeze(2)
    return preds


@torch.no_grad()
def similarity_percent(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Percent similarity to reference stems (higher is better),
    with explicit penalty for extra energy where the reference is (almost) silent.

    Bazuje na znormalizowanym MSE względem energii celu, ale dodaje karę
    za "nadmiarowy" sygnał tam, gdzie target jest bliski zera.

    preds, targets: (B, S, 1, L)
    Zwraca: % podobieństwa w [0, 100].
    """
    preds, targets = _match_time(preds, targets)

    # Podstawowe MSE / Var na amplitudach
    mse = torch.mean((preds - targets) ** 2)
    var = torch.mean(targets ** 2) + 1e-8

    # Nadmiarowa energia tam, gdzie target jest prawie cichy
    amp_t = targets.abs()
    amp_p = preds.abs()

    # Normalizacja względem maksymalnej amplitudy w targetach (per batch, źródło)
    max_t = amp_t.amax(dim=-1, keepdim=True).clamp(min=1e-6)  # (B, S, 1, 1)
    t_norm = amp_t / max_t  # [0, 1]
    p_norm = amp_p / max_t

    # Próg "ciszy" w targetach: wszystko poniżej thr_sil traktujemy jako ciszę/tło
    thr_sil = 0.1
    mask_silence = (t_norm < thr_sil).to(torch.float32)

    # Średnia nadmiarowa energia predykcji w miejscach ciszy (po normalizacji)
    excess = (mask_silence * (p_norm ** 2)).mean()

    # Waga kary za nadmiar (można dostroić w razie potrzeby)
    alpha = 0.5
    error = mse + alpha * excess

    sim = torch.clamp(1.0 - error / var, 0.0, 1.0) * 100.0
    return float(sim.item())


def pairwise_si_sdr(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    preds, targets = _match_time(preds, targets)
    if preds.dim() == 4:
        preds = preds.squeeze(2)
    if targets.dim() == 4:
        targets = targets.squeeze(2)
    L = min(preds.shape[-1], targets.shape[-1])
    preds = preds[..., :L]
    targets = targets[..., :L]
    dot = torch.sum(preds * targets, dim=-1)
    target_energy = torch.sum(targets ** 2, dim=-1) + eps
    scale = dot / target_energy
    proj = scale.unsqueeze(-1) * targets
    noise = preds - proj
    ratio = torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)


def si_sdr_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    si_sdr_vals = pairwise_si_sdr(preds, targets)
    return -si_sdr_vals.mean()


def pit_si_sdr_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds, targets = _match_time(preds, targets)
    if preds.dim() == 4:
        preds = preds.squeeze(2)
    if targets.dim() == 4:
        targets = targets.squeeze(2)
    B, S, L = preds.shape
    perms = list(itertools.permutations(range(S)))
    losses = []
    for perm in perms:
        permuted_preds = preds[:, perm, :]
        losses.append(si_sdr_loss(permuted_preds, targets))
    return torch.stack(losses).min()


def mrstft_loss(preds: torch.Tensor, targets: torch.Tensor,
                fft_sizes=(1024, 2048, 4096),
                hop_factors=(0.5, 0.25, 0.125),
                chunk_size: int | None = None) -> torch.Tensor:
    preds, targets = _match_time(preds, targets)
    if preds.dim() == 4:
        preds = preds.squeeze(2)
    if targets.dim() == 4:
        targets = targets.squeeze(2)
    device = preds.device
    B, S, L = preds.shape
    losses = []
    flat_preds = preds.reshape(B * S, L)
    flat_targets = targets.reshape(B * S, L)
    if chunk_size is None or chunk_size <= 0:
        chunk_size = flat_preds.size(0)

    for fft, hop_factor in zip(fft_sizes, hop_factors):
        hop = max(1, int(fft * hop_factor))
        window = torch.hann_window(fft, device=device)
        chunk_losses = []
        for start in range(0, flat_preds.size(0), chunk_size):
            end = start + chunk_size
            spec_pred = torch.stft(flat_preds[start:end], n_fft=fft, hop_length=hop, window=window, center=False, return_complex=True)
            spec_tgt = torch.stft(flat_targets[start:end], n_fft=fft, hop_length=hop, window=window, center=False, return_complex=True)
            chunk_loss = torch.mean(torch.abs(spec_pred.abs() - spec_tgt.abs()))
            chunk_losses.append(chunk_loss)
        losses.append(torch.stack(chunk_losses).mean())
    return torch.stack(losses).mean()
