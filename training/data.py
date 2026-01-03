import random
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class MusdbRandomChunks(Dataset):
    """
    Dataset returning random segments from MUSDB18.
    - sources: e.g. ["vocals", "drums", "bass", "other"]
    - mono: averages stereo channels to mono (simplification for a start)
    - segment length in seconds: segment_seconds

    Requires packages: musdb, stempeg
    """

    def __init__(
            self,
            root: str = "./musdb18-wav",
            subset: str = "train",
            sources: List[str] = ("vocals", "drums", "bass", "other"),
            sample_rate: int = 44100,
            segment_seconds: float = 6.0,
            items_per_epoch: int = 2000,
            mono: bool = True,
            seed: int = 42,
    ) -> None:
        super().__init__()
        try:
            import musdb  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "The 'musdb' package is not installed. Add to requirements: musdb, stempeg"
            ) from e

        # Load MUSDB
        import musdb
        self.db = musdb.DB(
            root=root,
            subsets=[subset],
            is_wav=True
        )
        self.tracks = list(iter(self.db))
        if len(self.tracks) == 0:
            raise RuntimeError(
                f"No tracks found in {root} for subset={subset}. "
            )

        self.sources = list(sources)
        self.sample_rate = sample_rate
        self.segment_seconds = segment_seconds
        self.segment_samples = int(segment_seconds * sample_rate)
        self.items_per_epoch = int(items_per_epoch)
        self.mono = mono
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        # Control number of samples per epoch to be independent from original track lengths
        return self.items_per_epoch

    def _ensure_channels_first(self, x: np.ndarray) -> np.ndarray:
        """Return array in (C, L) layout. Accepts (L, C) or (C, L) or (L,).
        Forces dtype float32.
        """
        if x.ndim == 1:
            # (L,) -> (1, L)
            x = x.astype(np.float32, copy=False)
            return x[None, :]
        # We expect one of two shapes: (samples, channels) or (channels, samples)
        # If the first dimension is very small (<= 4), it's likely (C, L)
        if x.shape[0] <= 4 and x.shape[1] > x.shape[0]:
            x = x.astype(np.float32, copy=False)
            return x
        # Otherwise assume (L, C) and transpose
        x = x.astype(np.float32, copy=False)
        return np.transpose(x, (1, 0))  # (C, L)

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        # x: (C, L)
        if self.mono:
            x = x.mean(axis=0, keepdims=True)  # (1, L)
        return torch.from_numpy(x.copy())  # float32

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Select a random track and a random start time
        track = self.rng.choice(self.tracks)

        track_length_samples = int(track.duration * self.sample_rate)

        start_sample = 0

        if track_length_samples > self.segment_samples:
            start_sample = self.rng.randint(0, track_length_samples - self.segment_samples)

        track.chunk_start = start_sample / self.sample_rate
        track.chunk_duration = self.segment_seconds

        target_dict: Dict[str, torch.Tensor] = {}
        for src in self.sources:
            audio = track.sources[src].audio
            audio = self._ensure_channels_first(audio)

            L_here = audio.shape[1]
            if L_here < self.segment_samples:
                pad = self.segment_samples - L_here
                audio = np.pad(audio, ((0, 0), (0, pad)))

            if audio.shape[1] > self.segment_samples:
                audio = audio[:, :self.segment_samples]

            target_dict[src] = self._to_tensor(audio)  # (C, L)

        # Build mixture as sum of all requested stems
        stacked = torch.stack(list(target_dict.values()), dim=0)  # (S, C, L)
        mixture = stacked.sum(dim=0)  # (C, L)

        return {
            "mixture": mixture,
            "targets": target_dict,
        }
