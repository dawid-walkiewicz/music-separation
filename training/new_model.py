import torch
import torch.nn as nn
import torch.nn.functional as F


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
