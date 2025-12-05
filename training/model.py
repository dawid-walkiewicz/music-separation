import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


class UNet2d(nn.Module):

    def __init__(self, n_class):
        super(UNet2d, self).__init__()
