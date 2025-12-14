import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

import argparse
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.unet2d.model import Unet2DWrapper
from training.data import MusdbRandomChunks

DEFAULT_SOURCES = ["vocals", "drums", "bass", "other"]
DEFAULT_N_FFT = 4096
DEFAULT_HOP_LENGTH = 1024


def _ensure_mono(wave: np.ndarray) -> np.ndarray:
    """Collapse any channel dimension to mono while keeping time dimension intact."""
    arr = np.asarray(wave)
    return arr.mean(axis=-2, keepdims=True) if arr.ndim >= 2 else arr


def load_model(ckpt_path: Path, stem_names: Sequence[str], device: torch.device) -> Unet2DWrapper:
    state = torch.load(str(ckpt_path), map_location="cpu")
    if "model" not in state:
        raise RuntimeError(f"Checkpoint {ckpt_path} does not contain 'model' weights")
    model = Unet2DWrapper(stem_names=list(stem_names))
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model


def separate_sources(model: Unet2DWrapper, mixture: torch.Tensor, device: torch.device) -> np.ndarray:
    mixture = mixture.to(device)
    with torch.no_grad():
        out = model.separate(mixture)
    stem_names = list(model.stems.keys())
    preds = [out[name].cpu().numpy() for name in stem_names]  # (2, L)
    preds = np.stack(preds, axis=0)  # (S, 2, L)
    preds = preds.mean(axis=1, keepdims=True)  # (S, 1, L)
    return preds


def _stack_targets(targets_obj, stem_names: Sequence[str]) -> torch.Tensor:
    if isinstance(targets_obj, dict):
        return torch.stack([targets_obj[name] for name in stem_names], dim=0)
    return targets_obj


def _log_spectrogram(wave: np.ndarray, sr: int, n_fft: int, hop: int, window: torch.Tensor) -> np.ndarray:
    wave = np.asarray(wave)
    wave = np.squeeze(wave)
    if wave.ndim != 1:
        raise ValueError(f"Waveform must reduce to 1D, got shape {wave.shape}")
    tensor = torch.from_numpy(wave).float()
    spec = torch.stft(tensor, n_fft=n_fft, hop_length=hop, window=window, center=True, return_complex=True)
    mag = spec.abs().numpy()
    return np.log10(np.maximum(mag, 1e-8))


def plot_stack(mixture: np.ndarray,
               targets: np.ndarray,
               predictions: np.ndarray,
               stem_names: Sequence[str],
               sr: int,
               n_fft: int,
               hop: int,
               window: torch.Tensor,
               title: str,
               out_path: Path):
    mixture = np.asarray(mixture)
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)

    common_len = min(mixture.shape[-1], targets.shape[-1], predictions.shape[-1])
    mixture = mixture[..., :common_len]
    targets = targets[..., :common_len]
    predictions = predictions[..., :common_len]

    mix_spec = _log_spectrogram(mixture, sr, n_fft, hop, window)
    duration = common_len / sr
    freq_max = sr / 2

    rows = len(stem_names)
    fig = plt.figure(figsize=(12, 3 * (rows + 0.5)), constrained_layout=True)
    gs = fig.add_gridspec(rows + 1, 2)

    mix_ax = fig.add_subplot(gs[0, :])
    mix_ax.imshow(mix_spec, origin="lower", aspect="auto", extent=(0.0, duration, 0.0, freq_max), cmap="magma")
    mix_ax.set_ylabel("Mixture")
    mix_ax.set_title("Mixture (input)")
    mix_ax.set_xticklabels([])

    axes_for_cb: list[plt.Axes] = []
    last_im = None

    for row, stem_name in enumerate(stem_names):
        ax_tgt = fig.add_subplot(gs[row + 1, 0], sharex=mix_ax, sharey=mix_ax)
        ax_pred = fig.add_subplot(gs[row + 1, 1], sharex=mix_ax, sharey=mix_ax)
        axes_for_cb.extend([ax_tgt, ax_pred])

        tgt = targets[row:row + 1]
        pred = predictions[row:row + 1]
        tgt_spec = _log_spectrogram(tgt, sr, n_fft, hop, window)
        pred_spec = _log_spectrogram(pred, sr, n_fft, hop, window)

        last_im = ax_tgt.imshow(tgt_spec, origin="lower", aspect="auto", extent=(0.0, duration, 0.0, freq_max), cmap="magma")
        ax_pred.imshow(pred_spec, origin="lower", aspect="auto", extent=(0.0, duration, 0.0, freq_max), cmap="magma")

        ax_tgt.set_ylabel(stem_name)
        if row == 0:
            ax_tgt.set_title("Target")
            ax_pred.set_title("Prediction")
        if row == rows - 1:
            ax_pred.set_xlabel("Time [s]")
            ax_tgt.set_xlabel("Time [s]")

    fig.suptitle(title)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes_for_cb, fraction=0.02, pad=0.02, label="log10 |STFT|")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_demo(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = load_model(ckpt_path, DEFAULT_SOURCES, device)

    dataset = MusdbRandomChunks(
        root=args.data_root,
        subset="train",
        segment_seconds=args.segment_seconds,
        items_per_epoch=args.samples,
        mono=False,
        seed=args.seed,
    )

    window = torch.hann_window(DEFAULT_N_FFT)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(args.samples):
        sample = dataset[idx]
        mixture = sample["mixture"].clone()
        targets = _stack_targets(sample["targets"], DEFAULT_SOURCES).clone()
        preds = separate_sources(model, mixture, device)

        mixture_np = mixture.numpy()
        targets_np = targets.numpy()

        mixture_np = _ensure_mono(mixture_np)
        targets_np = _ensure_mono(targets_np)

        print(f"Saving spectrograms for sample {idx + 1}/{args.samples}")
        title = f"Sample {idx + 1}"
        out_path = out_dir / f"sample_{idx + 1:03d}.png"
        plot_stack(mixture_np, targets_np, preds, DEFAULT_SOURCES, dataset.sample_rate, DEFAULT_N_FFT, DEFAULT_HOP_LENGTH, window, title, out_path)


def build_parser():
    parser = argparse.ArgumentParser(description="Generate spectrogram comparisons for MUSDB samples and model predictions.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_root", type=str, default="./musdb18-wav", help="MUSDB18 data root directory")
    parser.add_argument("--output_dir", type=str, default="./demonstration/spectrograms", help="Directory to save spectrograms")
    parser.add_argument("--samples", type=int, default=2, help="Number of random samples to visualize")
    parser.add_argument("--segment_seconds", type=float, default=6.0, help="Segment length in seconds")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for sample selection")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
