from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor


def ensure_stereo(mixture: Tensor) -> tuple[Tensor, int, int]:
    """
    Check and ensure that the input audio mixture has two channels (stereo).

    Args:
        mixture (torch.Tensor): Input audio tensor of shape (B, C, L)

    Returns:
        mixture (torch.Tensor): Mixture in stereo.
        B (int): Batch size.
        L (int): Length of the audio signal.
    """
    if mixture.dim() != 3:
        raise ValueError(f"Expected mixture of shape (B, C, L), got {mixture.shape}")
    B, C, L = mixture.shape
    if C == 1:
        mixture_stereo = mixture.repeat(1, 2, 1)  # (B, 2, L)
    elif C == 2:
        mixture_stereo = mixture
    else:
        raise ValueError(f"Splitter expects 1 or 2 channels, got C={C}")

    return mixture_stereo, B, L


def separate_file(
        model: torch.nn.Module,
        read_audio: Callable[[Path, int], np.ndarray],
        target_sr: int,
        separate_chunked: Callable[[torch.nn.Module, np.ndarray, int, torch.device, float, float, bool], np.ndarray],
        sources_list: tuple[str, str, str, str],
        write_audio: Callable[[Path, np.ndarray, int], None],
        path: Path,
        out_dir: Path,
        device: torch.device,
        chunk_seconds: float,
        overlap_seconds: float,
        amp: bool
):
    name = path.stem
    print(f"Processing: {path} -> {out_dir / name}")
    wav = read_audio(path, target_sr)
    wav = wav.astype(np.float32)
    # Chunked separation
    try:
        sources = separate_chunked(model, wav, target_sr, device, chunk_seconds, overlap_seconds, amp)  # (S, L)
    except RuntimeError as e:
        # Fallback to CPU if CUDA OOM
        if "CUDA out of memory" in str(e) and device.type == "cuda":
            print("CUDA OOM encountered. Falling back to CPU for this file...")
            cpu_device = torch.device("cpu")
            sources = separate_chunked(model.to(cpu_device), wav, target_sr, cpu_device, chunk_seconds, overlap_seconds,
                                       amp)
            model.to(device)  # move back
        else:
            raise
    # Save per source
    for i, src in enumerate(sources_list):
        out_path = out_dir / name / f"{src}.wav"
        data = sources[i]
        maxv = float(np.max(np.abs(data)))
        if maxv > 0:
            data = data / maxv * 0.99
        write_audio(out_path, data, target_sr)


def process_input(
        model: torch.nn.Module,
        read_audio: Callable[[Path, int], np.ndarray],
        target_sr: int,
        separate_chunked: Callable[[torch.nn.Module, np.ndarray, int, torch.device, float, float, bool], np.ndarray],
        sources: tuple[str, str, str, str],
        write_audio: Callable[[Path, np.ndarray, int], None],
        input_path: Path,
        out_root: Path,
        device: torch.device,
        chunk_seconds: float,
        overlap_seconds:
        float,
        amp: bool
):
    if input_path.is_dir():
        exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        files = [p for p in sorted(input_path.iterdir()) if (p.suffix.lower() in exts)]
        if not files:
            print(f"No audio files found in {input_path}")
            return
        for p in files:
            separate_file(model, read_audio, target_sr, separate_chunked, sources, write_audio, p, out_root, device,
                          chunk_seconds, overlap_seconds, amp)
    elif input_path.is_file():
        separate_file(model, read_audio, target_sr, separate_chunked, sources, write_audio, input_path, out_root,
                      device, chunk_seconds, overlap_seconds, amp)
    else:
        raise RuntimeError(f"Input path not found: {input_path}")
