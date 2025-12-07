import torch
import torch.nn as nn
import torch.nn.functional as F
#https://docs.pytorch.org/docs/stable/generated/torch.stft.html
def compute_log_power_spectrogram(waveform, n_fft=2048, hop_length=512, win_length=None, eps=1e-10):
    if win_length is None:
        win_length = n_fft
    window = torch.hann_window(win_length, device=waveform.device)
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
    magnitude = stft.abs()
    power = magnitude ** 2
    log_power = 10.0 * torch.log10(power + eps)
    return log_power

class NewSeparationModel(nn.Module):
    """Lightweight placeholder so training can instantiate the future model."""

    def __init__(self, n_sources: int, base: int = 32):
        super().__init__()
        self.n_sources = n_sources
        self.enc = nn.Conv1d(1, base, kernel_size=3, padding=1)
        self.dec = nn.Conv1d(base, n_sources, kernel_size=1)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.enc(mixture))
        masks = torch.softmax(self.dec(x), dim=1)
        return masks

