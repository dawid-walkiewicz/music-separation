import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, k, s, p)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


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
        self.u3 = nn.ConvTranspose1d(ch * 8, ch * 4, 4, 2, 1)
        self.p3 = nn.Sequential(ConvBlock(ch * 8, ch * 4), ConvBlock(ch * 4, ch * 4))

        self.u2 = nn.ConvTranspose1d(ch * 4, ch * 2, 4, 2, 1)
        self.p2 = nn.Sequential(ConvBlock(ch * 4, ch * 2), ConvBlock(ch * 2, ch * 2))

        self.u1 = nn.ConvTranspose1d(ch * 2, ch, 4, 2, 1)
        self.p1 = nn.Sequential(ConvBlock(ch * 2, ch), ConvBlock(ch, ch))

        # Output masks
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
        x1 = self.e1(mixture)  # (B, ch, L)
        x2 = self.e2(F.leaky_relu(self.d1(x1), 0.1))  # (B, 2ch, L/2)
        x3 = self.e3(F.leaky_relu(self.d2(x2), 0.1))  # (B, 4ch, L/4)
        xb = self.b(F.leaky_relu(self.d3(x3), 0.1))   # (B, 8ch, L/8)

        y3 = self.u3(xb)  # (B, 4ch, ~L/4)

        y3 = torch.cat([y3, x3], dim=1)
        y3 = self.p3(y3)

        y2 = self.u2(y3)  # (B, 2ch, ~L/2)
        y2 = torch.cat([y2, x2], dim=1)
        y2 = self.p2(y2)

        y1 = self.u1(y2)  # (B, ch, ~L)
        y1 = torch.cat([y1, x1], dim=1)
        y1 = self.p1(y1)

        masks = self.act_out(self.out(y1))  # (B, S, ~L)
        return masks


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
    """Simple frequency-domain branch operating on STFT magnitude.

    Input:  magnitude STFT (B, 1, F, T)
    Output: 1D time features (B, C, T) after pooling over frequency.
    """

    def __init__(self, in_ch: int = 1, base: int = 32, out_ch: int | None = None):
        super().__init__()
        if out_ch is None:
            out_ch = base
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(base * 4, out_ch, kernel_size=1)

    def forward(self, spec_mag: torch.Tensor) -> torch.Tensor:
        # spec_mag: (B, 1, F, T)
        x = self.enc1(spec_mag)
        x = self.enc2(x)
        x = self.enc3(x)  # (B, C, F', T')
        x = self.proj(x)  # (B, out_ch, F', T')
        # global pooling over frequency F'
        x = x.mean(dim=2)  # (B, out_ch, T')
        return x


class HybridTimeFreqUNet(nn.Module):
    """Hybrid model combining time-domain UNet1D with a simple STFT-based branch.

    - Input:  mixture (B, 1, L)
    - Output: masks (B, S, Lt)

    Steps:
    1. UNet1D generates time-domain masks.
    2. STFT + FreqBranch generate frequency features.
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
        self.freq_branch = FreqBranch(in_ch=1, base=base_freq, out_ch=base_freq)

        # Fusion layer: concatenates time-domain masks (S channels)
        # with frequency-domain features (base_freq channels)
        self.fuse_conv = nn.Conv1d(n_sources + base_freq, n_sources, kernel_size=1)

        # Cached Hann window for STFT; automatically moved to correct device
        self.register_buffer("stft_window", torch.hann_window(self.n_fft))

    def _maybe_checkpoint(self, module: nn.Module, *args: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return grad_checkpoint(module, *args)
        return module(*args)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        # mixture: (B, 1, L)
        B, C, L = mixture.shape

        # 1) Time-domain branch
        masks_time = self._maybe_checkpoint(self.time_net, mixture)  # (B, S, Lt)

        # 2) Frequency-domain branch via STFT
        wav = mixture.squeeze(1)  # (B, L)
        window = self.stft_window.to(wav.device)
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )  # (B, F, T)
        mag = spec.abs().unsqueeze(1)  # (B, 1, F, T)

        freq_feats = self._maybe_checkpoint(self.freq_branch, mag)  # (B, base_freq, T')

        # 3) Match time length: interpolate frequency features to Lt
        Lt = masks_time.shape[-1]
        freq_feats_up = F.interpolate(freq_feats, size=Lt, mode="linear", align_corners=False)

        # 4) Fusion and final masks
        x = torch.cat([masks_time, freq_feats_up], dim=1)  # (B, S+base_freq, Lt)
        masks = torch.sigmoid(self.fuse_conv(x))  # (B, S, Lt)
        return masks


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


