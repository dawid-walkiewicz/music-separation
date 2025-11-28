from __future__ import annotations
import argparse
from pathlib import Path
import sys
import torch
import numpy as np

# Ensure project root on sys.path for 'training' imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local imports
from training.model import UNet1D, apply_masks

# Try backends for audio IO
_have_soundfile = False
_have_torchaudio = False
_have_resampy = False
_have_stempeg = False
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
try:
    import stempeg
    _have_stempeg = True
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


def _forward_sources(model: torch.nn.Module, wav_mono: np.ndarray, device: torch.device, amp: bool = True) -> np.ndarray:
    """Run model on a mono waveform and return sources as np array (S, Lout)."""
    assert wav_mono.ndim == 1
    x = torch.from_numpy(wav_mono[None, None, :].astype(np.float32)).to(device)
    with torch.no_grad():
        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                masks = model(x)  # (1, S, Lm)
        else:
            masks = model(x)
        sources = apply_masks(x, masks)  # (1, S, 1, Lc)
    sources = sources[0, :, 0].detach().cpu().numpy()  # (S, Lc)
    return sources


def separate_chunked(model: torch.nn.Module,
                     wav_mono: np.ndarray,
                     sr: int,
                     device: torch.device,
                     chunk_seconds: float = 8.0,
                     overlap_seconds: float = 1.0,
                     amp: bool = True) -> np.ndarray:
    """Chunked separation with overlap-add to avoid GPU OOM and seam artifacts.

    Returns: np.ndarray shape (S, L)
    """
    L = int(wav_mono.shape[0])
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
        chunk_wav = wav_mono[pos:end]
        # Forward
        pred = _forward_sources(model, chunk_wav, device, amp)  # (S, M)
        M = pred.shape[1]
        # Create window of length M (Hann), avoid zeros at exact edges for averaging stability on tiny chunks
        if M > 1:
            win = np.hanning(M).astype(np.float32)
            # If chunk not overlapped (e.g., first/last), ensure non-zero weights
            if pos == 0:
                win[: M // 4] = np.maximum(win[: M // 4], 0.5)
            if end == L:
                win[-(M // 4):] = np.maximum(win[-(M // 4):], 0.5)
        else:
            win = np.ones(M, dtype=np.float32)
        # Accumulate
        out[:, pos:pos + M] += pred * win[None, :]
        weight[pos:pos + M] += win
        pos += hop

    # Normalize by weights to combine overlaps
    nz = weight > 0
    out[:, nz] /= weight[nz][None, :]
    # Fill any untouched samples (edge cases) by simple copy from nearest (optional)
    if not np.all(nz):
        idx = np.where(~nz)[0]
        for i in idx:
            j = i - 1 if i > 0 else i + 1
            out[:, i] = out[:, j]
    return out


def read_audio(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Read audio file and return mono float32 numpy array at target_sr.

    Returns shape (L,) with values in float32.
    Supports: wav/mp3/flac/ogg/m4a via soundfile/torchaudio, and MUSDB .stem.mp4 via stempeg.
    """
    path = Path(path)
    ext = path.suffix.lower()

    # Special case: MUSDB stem file
    if ext == ".mp4" and path.name.endswith(".stem.mp4"):
        if not _have_stempeg:
            raise RuntimeError("Reading .stem.mp4 requires `stempeg`. Install it and try again.")
        # stems: (S, T, C); typical S=5 with [mixture, vocals, drums, bass, other]
        stems, sr = stempeg.read_stems(str(path), dtype=np.float32)
        # Prefer explicit mixture if present (S>=1), else sum sources
        if stems.ndim == 3 and stems.shape[0] >= 1:
            mix = stems[0]  # (T, C)
        else:
            mix = stems.sum(axis=0)  # (T, C)
        if mix.ndim == 2 and mix.shape[1] > 1:
            mix_mono = np.mean(mix, axis=1)
        else:
            mix_mono = mix.squeeze()
        wave = _resample_np_mono(mix_mono, sr, target_sr)
        return wave.astype(np.float32, copy=False)

    # Generic audio via soundfile
    if _have_soundfile:
        data, sr = sf.read(str(path), always_2d=True)
        data = data.T  # (C, L)
        if data.shape[0] > 1:
            data = np.mean(data, axis=0, keepdims=True)
        data = data.squeeze(0)
        data = _resample_np_mono(data, sr, target_sr)
        return data.astype(np.float32, copy=False)

    # Fallback via torchaudio
    if _have_torchaudio:
        wav, sr = torchaudio.load(str(path))  # (C, L)
        wav = wav.mean(dim=0)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), orig_freq=sr, new_freq=target_sr)
            wav = wav.squeeze(0)
        return wav.numpy().astype(np.float32)

    raise RuntimeError("No audio backend available: install soundfile (pysoundfile) or torchaudio or stempeg for .stem.mp4")


def write_audio(path: Path, data: np.ndarray, sr: int = TARGET_SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    # data: 1D numpy
    if _have_soundfile:
        sf.write(str(path), data, sr)
        return
    if _have_torchaudio:
        wav = torch.from_numpy(data).unsqueeze(0)
        torchaudio.save(str(path), wav, sample_rate=sr)
        return
    raise RuntimeError("No audio backend available for writing: install soundfile or torchaudio")


def load_model_from_ckpt(ckpt_path: Path, device: torch.device):
    # Load checkpoint safely; newer torch may support weights_only to avoid pickle execution
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        # older torch doesn't have weights_only arg
        ckpt = torch.load(str(ckpt_path), map_location="cpu")

    # Support checkpoints that store the state dict under the 'model' key or are the state dict directly
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

    n_sources = len(SOURCES)
    model = UNet1D(n_sources=n_sources, base=64)

    # Try strict load first
    try:
        model.load_state_dict(state)
        print("Checkpoint loaded (strict match).")
    except RuntimeError as e:
        print(f"Warning: strict state_dict load failed: {e}\nTrying to auto-remap keys to match current model...")
        # Compute expected/present keys
        expected_keys = set(model.state_dict().keys())
        present_keys = set(state.keys()) if isinstance(state, dict) else set()
        missing = expected_keys - present_keys
        unexpected = present_keys - expected_keys

        # Heuristic: try to map unexpected keys to missing keys by adjusting numeric tokens by +/-1
        mapping = {}
        for u in list(unexpected):
            toks = u.split('.')
            for idx, tok in enumerate(toks):
                if tok.isdigit():
                    for delta in (-1, 1):
                        try:
                            new_tok = str(int(tok) + delta)
                        except Exception:
                            continue
                        new_toks = toks.copy()
                        new_toks[idx] = new_tok
                        cand = '.'.join(new_toks)
                        if cand in missing:
                            mapping[u] = cand
                            break
                # also try common case: replace '.2.'->'.1.', '.3.'->'.2.' etc by searching for numeric substrings
                if any(char.isdigit() for char in tok) and tok != toks[idx]:
                    pass
                if u in mapping:
                    break

        if mapping:
            print("Attempting auto-mapping of keys (this may fix small layer-index shifts):")
            for k, v in mapping.items():
                print(f"  {k} -> {v}")
            # Apply mapping to a copy of state
            new_state = dict(state)
            for src_key, dst_key in mapping.items():
                new_state[dst_key] = new_state[src_key]
                del new_state[src_key]
            # Try strict load with remapped keys
            try:
                model.load_state_dict(new_state)
                print("Checkpoint loaded after auto-remapping (strict match).")
            except Exception as e2:
                print(f"Auto-remap didn't produce a strict match: {e2}\nFalling back to non-strict load (missing/unexpected keys will be ignored).")
                load_res = model.load_state_dict(state, strict=False)
                missing = getattr(load_res, "missing_keys", None) or (load_res.get("missing_keys") if isinstance(load_res, dict) else None)
                unexpected = getattr(load_res, "unexpected_keys", None) or (load_res.get("unexpected_keys") if isinstance(load_res, dict) else None)
                if missing:
                    print("Missing keys when loading state_dict:", missing)
                if unexpected:
                    print("Unexpected keys in state_dict (ignored):", unexpected)
        else:
            print("No useful remapping found. Falling back to non-strict load (missing/unexpected keys will be ignored).")
            load_res = model.load_state_dict(state, strict=False)
            missing = getattr(load_res, "missing_keys", None) or (load_res.get("missing_keys") if isinstance(load_res, dict) else None)
            unexpected = getattr(load_res, "unexpected_keys", None) or (load_res.get("unexpected_keys") if isinstance(load_res, dict) else None)
            if missing:
                print("Missing keys when loading state_dict:", missing)
            if unexpected:
                print("Unexpected keys in state_dict (ignored):", unexpected)

    model.to(device)
    model.eval()
    return model


def separate_file(model: torch.nn.Module, path: Path, out_dir: Path, device: torch.device,
                  chunk_seconds: float, overlap_seconds: float, amp: bool):
    name = path.stem
    print(f"Processing: {path} -> {out_dir / name}")
    wav = read_audio(path, TARGET_SR)
    wav = wav.astype(np.float32)
    # Chunked separation
    try:
        sources = separate_chunked(model, wav, TARGET_SR, device, chunk_seconds, overlap_seconds, amp)  # (S, L)
    except RuntimeError as e:
        # Fallback to CPU if CUDA OOM
        if "CUDA out of memory" in str(e) and device.type == "cuda":
            print("CUDA OOM encountered. Falling back to CPU for this file...")
            cpu_device = torch.device("cpu")
            sources = separate_chunked(model.to(cpu_device), wav, TARGET_SR, cpu_device, chunk_seconds, overlap_seconds, amp)
            model.to(device)  # move back
        else:
            raise
    # Save per source
    for i, src in enumerate(SOURCES):
        out_path = out_dir / name / f"{src}.wav"
        data = sources[i]
        maxv = float(np.max(np.abs(data)))
        if maxv > 0:
            data = data / maxv * 0.99
        write_audio(out_path, data, TARGET_SR)


def process_input(model: torch.nn.Module, input_path: Path, out_root: Path, device: torch.device,
                  chunk_seconds: float, overlap_seconds: float, amp: bool):
    if input_path.is_dir():
        exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".stem.mp4"}
        files = [p for p in sorted(input_path.iterdir()) if (p.suffix.lower() in exts) or p.name.endswith(".stem.mp4")]
        if not files:
            print(f"No audio files found in {input_path}")
            return
        for p in files:
            separate_file(model, p, out_root, device, chunk_seconds, overlap_seconds, amp)
    elif input_path.is_file():
        separate_file(model, input_path, out_root, device, chunk_seconds, overlap_seconds, amp)
    else:
        raise RuntimeError(f"Input path not found: {input_path}")


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
    process_input(model, input_path, out_root, device, args.chunk_seconds, args.overlap_seconds, use_amp)


if __name__ == "__main__":
    main()
