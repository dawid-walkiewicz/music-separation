import os
import random
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
import soundfile as sf


class MusdbRandomChunks(Dataset):
    """
    Dataset zwracający losowe segmenty z MUSDB18.
    - źródła: np. ["vocals", "drums", "bass", "other"]
    - mono: uśrednia kanały stereo do mono (uproszczenie na start)
    - długość segmentu w sekundach: segment_seconds

    Wymaga pakietów: musdb, stempeg
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
                "Pakiet 'musdb' nie jest zainstalowany. Dodaj do requirements: musdb, stempeg"
            ) from e

        # Ładujemy MUSDB
        import musdb
        self.db = musdb.DB(
            root=root,
            subsets=[subset],
            is_wav=True,  # .stem.mp4
        )
        self.tracks = list(iter(self.db))
        if len(self.tracks) == 0:
            raise RuntimeError(
                f"Nie znaleziono utworów w {root} dla subset={subset}. Upewnij się, że struktura MUSDB18 jest poprawna."
            )

        self.sources = list(sources)
        self.sample_rate = sample_rate
        self.segment_seconds = segment_seconds
        self.segment_samples = int(segment_seconds * sample_rate)
        self.items_per_epoch = int(items_per_epoch)
        self.mono = mono
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        # Kontrolujemy liczbę próbek na epokę, aby uniezależnić się od długości oryginalnych utworów
        return self.items_per_epoch

    def _ensure_channels_first(self, x: np.ndarray) -> np.ndarray:
        """Zwraca tablicę w układzie (C, L). Akceptuje (L, C) lub (C, L) lub (L,).
        Wymusza dtype float32.
        """
        if x.ndim == 1:
            # (L,) -> (1, L)
            x = x.astype(np.float32, copy=False)
            return x[None, :]
        # Spodziewamy się jednej z dwóch postaci: (samples, channels) lub (channels, samples)
        # Jeśli pierwszy wymiar jest bardzo mały (<= 4), prawdopodobnie (C, L)
        if x.shape[0] <= 4 and x.shape[1] > x.shape[0]:
            x = x.astype(np.float32, copy=False)
            return x
        # W przeciwnym razie przyjmij (L, C) i transponuj
        x = x.astype(np.float32, copy=False)
        return np.transpose(x, (1, 0))  # (C, L)

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        # x: (C, L)
        if self.mono:
            x = x.mean(axis=0, keepdims=True)  # (1, L)
        return torch.from_numpy(x.copy())  # float32

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Wybieramy losowy utwór i losowy start (po osi czasu)
        track = self.rng.choice(self.tracks)

        track_length_samples = int(track.duration * self.sample_rate)

        start_sample = 0

        if track_length_samples > self.segment_samples:
            start_sample = self.rng.randint(0, track_length_samples - self.segment_samples)

        track.chunk_start = start_sample / self.sample_rate
        track.chunk_duration = self.segment_seconds

        target_list = []
        for src in self.sources:
            source_file_path = str(track.sources[src].path)
            try:
                sf.info(source_file_path)
            except Exception as e:
                raise RuntimeError(
                    f"KRYTYCZNY BŁĄD: Plik WAV jest uszkodzony lub nieczytelny: {source_file_path}\n"
                    f"Utwór: {track.name}\n"
                    f"Oryginalny błąd 'soundfile': {e}\n"
                    f"Proszę naprawić konwersję tego pliku."
                )

            audio = track.sources[src].audio
            audio = self._ensure_channels_first(audio)

            L_here = audio.shape[1]
            if L_here < self.segment_samples:
                pad = self.segment_samples - L_here
                audio = np.pad(audio, ((0, 0), (0, pad)))

            if audio.shape[1] > self.segment_samples:
                audio = audio[:, :self.segment_samples]

            target_list.append(self._to_tensor(audio))  # (C, L)

        stacked = torch.stack(target_list, dim=0)
        mixture = stacked.sum(dim=0)

        targets = stacked

        return {
            "mixture": mixture,
            "targets": targets,
        }