from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from models.audio_functions import process_input
from models.unet2d.model import Unet2DWrapper

_have_soundfile = False
_have_torchaudio = False
_have_resampy = False
try:
    import soundfile as sf
    _have_soundfile = True
except Exception:
    pass
try:
    import torchaudio
    _have_torchaudio = True
except Exception:
    pass
try:
    import resampy
    _have_resampy = True
except Exception:
    pass


TARGET_SR = 44100
SOURCES = ("vocals", "drums", "bass", "other")


def _resample_np_mono(wave_mono: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return wave_mono.astype(np.float32, copy=False)
    if _have_resampy:
        return resampy.resample(wave_mono, sr, target_sr).astype(np.float32, copy=False)
    if _have_torchaudio:
        wav = torch.from_numpy(wave_mono).unsqueeze(0)
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
        return wav.squeeze(0).numpy().astype(np.float32, copy=False)
    raise RuntimeError("Need resampling support: install `resampy` or `torchaudio` to resample audio")


def read_audio(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Read audio file and return mono or stereo float32 numpy array at target_sr.

    Returns shape (C, L) with values in float32. C can be 1 or 2.
    Supports: wav/mp3/flac/ogg/m4a via soundfile/torchaudio.
    """
    path = Path(path)

    if _have_soundfile:
        data, sr = sf.read(str(path), always_2d=True)  # (L, C)
        data = data.T  # (C, L)
        if sr != target_sr:
            if data.shape[0] > 1:
                res = []
                for ch in range(data.shape[0]):
                    res.append(_resample_np_mono(data[ch], sr, target_sr))
                data = np.stack(res, axis=0)
            else:
                data = _resample_np_mono(data.squeeze(0), sr, target_sr)[None, :]
        return data.astype(np.float32, copy=False)

    # Fallback via torchaudio
    if _have_torchaudio:
        wav, sr = torchaudio.load(str(path))  # (C, L)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
        return wav.numpy().astype(np.float32)

    raise RuntimeError("No audio backend available: install soundfile (pysoundfile) or torchaudio")


def write_audio(path: Path, data: np.ndarray, sr: int = TARGET_SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    # data: (L,) or (C, L)
    if data.ndim == 1:
        data_to_write = data
    else:
        data_to_write = data.T  # (L, C)
    if _have_soundfile:
        sf.write(str(path), data_to_write, sr)
        return
    if _have_torchaudio:
        wav = torch.from_numpy(data).float()
        torchaudio.save(str(path), wav, sample_rate=sr)
        return
    raise RuntimeError("No audio backend available for writing: install soundfile or torchaudio")


def _ensure_stereo(wav: np.ndarray) -> np.ndarray:
    """Ensure waveform is stereo: input (C, L) -> (2, L)."""
    if wav.ndim != 2:
        raise ValueError(f"Expected audio with shape (C, L), got {wav.shape}")
    C, L = wav.shape
    if C == 2:
        return wav
    if C == 1:
        return np.tile(wav, (2, 1))
    # More than 2 channels: use first two
    return wav[:2]


def separate_chunked(model: torch.nn.Module,
                     wav: np.ndarray,
                     sr: int,
                     device: torch.device,
                     chunk_seconds: float = 8.0,
                     overlap_seconds: float = 1.0,
                     amp: bool = True) -> np.ndarray:
    """Chunked separation with overlap-add using Unet2DWrapper.

    Args:
        model:   Unet2DWrapper instance
        wav:     np.ndarray (C, L), mono or stereo
        sr:      sample rate
        device:  torch.device

    Returns:
        sources: np.ndarray of shape (S, L), mono mix-down of separated sources.
    """
    wav = _ensure_stereo(wav)  # (2, L)
    L = wav.shape[1]
    if L == 0:
        return np.zeros((len(SOURCES), 0), dtype=np.float32)

    chunk = max(int(chunk_seconds * sr), 1)
    overlap = max(int(overlap_seconds * sr), 0)
    hop = max(chunk - overlap, 1)

    out = np.zeros((len(SOURCES), L), dtype=np.float32)
    weight = np.zeros(L, dtype=np.float32)

    pos = 0
    while pos < L:
        end = min(pos + chunk, L)
        chunk_wav = wav[:, pos:end]  # (2, M)

        # To tensor on device
        chunk_tensor = torch.from_numpy(chunk_wav).to(device)

        with torch.no_grad():
            if amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    out_dict = model.separate(chunk_tensor)  # name -> (2, M_out)
            else:
                out_dict = model.separate(chunk_tensor)

        stem_names = list(model.stems.keys())
        stems = [out_dict[name].cpu().numpy() for name in stem_names]  # list of (2, M_out)
        stems = np.stack(stems, axis=0)  # (S, 2, M_out)
        # Downmix to mono for writing / evaluation
        pred = stems.mean(axis=1)  # (S, M_out)
        M = pred.shape[1]

        # Window for overlap-add
        if M > 1:
            win = np.hanning(M).astype(np.float32)
            if pos == 0:
                win[: M // 4] = np.maximum(win[: M // 4], 0.5)
            if end == L:
                win[-(M // 4):] = np.maximum(win[-(M // 4):], 0.5)
        else:
            win = np.ones(M, dtype=np.float32)

        out[:, pos:pos + M] += pred * win[None, :]
        weight[pos:pos + M] += win
        pos += hop

    nz = weight > 0
    out[:, nz] /= weight[nz][None, :]
    if not np.all(nz):
        idx = np.where(~nz)[0]
        for i in idx:
            j = i - 1 if i > 0 else i + 1
            out[:, i] = out[:, j]
    return out.astype(np.float32, copy=False)


def load_model_from_ckpt(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model = Unet2DWrapper(stem_names=list(SOURCES))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt saved by training.train")
    parser.add_argument("--input", type=str, required=True, help="Input file or directory")
    parser.add_argument("--out", type=str, default="usage/separated", help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device: cpu or cuda, default auto-detect")
    parser.add_argument("--chunk_seconds", type=float, default=8.0, help="Chunk length in seconds for inference")
    parser.add_argument("--overlap_seconds", type=float, default=1.0, help="Overlap between chunks in seconds")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision on CUDA to reduce risk of NaNs")
    args = parser.parse_args(argv)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    input_path = Path(args.input)
    out_root = Path(args.out)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_from_ckpt(ckpt_path, device)
    use_amp = (not args.no_amp)
    process_input(model, read_audio, TARGET_SR, separate_chunked, SOURCES, write_audio, input_path, out_root, device, args.chunk_seconds, args.overlap_seconds, use_amp)


if __name__ == "__main__":
    main()
