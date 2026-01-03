from __future__ import annotations

from pathlib import Path
from typing import List
from museval import metrics as muse_metrics
import argparse
import json

import numpy as np
import soundfile as sf


DEFAULT_TRACK_NAME = "Al James - Schoolboy Facination"

SOURCES: List[str] = ["vocals", "drums", "bass", "other"]


def load_wav_mono(path: Path) -> np.ndarray:
    """Load a WAV file and return a mono signal as a 1-D numpy array.

    Stereo files are downmixed by averaging channels.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sr = sf.read(path, always_2d=True)
    mono = audio.mean(axis=1)
    return mono.astype(np.float32)


def load_reference_sources(musdb_root: Path, track_name: str) -> np.ndarray:
    """Load reference sources for a single MUSDB18 track.

    Expected layout: <musdb_root>/test/<track_name>/{vocals,drums,bass,other}.wav
    Returns an array with shape (S, L) where S is number of sources and L is number of samples.
    All sources are trimmed to the same minimum length.
    """
    track_dir = musdb_root / "test" / track_name
    if not track_dir.is_dir():
        raise FileNotFoundError(f"MUSDB track directory not found: {track_dir}")

    refs = []
    for name in SOURCES:
        ref_path = track_dir / f"{name}.wav"
        if not ref_path.is_file():
            raise FileNotFoundError(
                f"Missing reference file for source '{name}': {ref_path}\n"
                f"Ensure MUSDB18-wav is extracted correctly."
            )
        refs.append(load_wav_mono(ref_path))

    min_len = min(ref.shape[0] for ref in refs)
    refs = [ref[:min_len] for ref in refs]

    return np.stack(refs, axis=0)


def load_estimated_sources(estimates_root: Path, track_name: str) -> np.ndarray:
    """Load estimated sources from estimates_root for a track.

    Supported layouts:
      - <estimates_root>/{vocals,drums,bass,other}.wav
      - <estimates_root>/<track_name>/{...}
    Returns an array with shape (S, L_est).
    """
    if estimates_root.is_file():
        raise FileNotFoundError(f"estimates_root points to a file, expected a directory: {estimates_root}")

    if (estimates_root / track_name).is_dir():
        est_dir = estimates_root / track_name
    else:
        est_dir = estimates_root

    if not est_dir.is_dir():
        raise FileNotFoundError(f"Estimates directory not found: {est_dir}")

    ests = []
    for name in SOURCES:
        est_path = est_dir / f"{name}.wav"
        if not est_path.is_file():
            raise FileNotFoundError(
                f"Missing estimate file for source '{name}': {est_path}\n"
                f"Ensure separation for this track has been performed and files exist."
            )
        ests.append(load_wav_mono(est_path))

    return np.stack(ests, axis=0)


def save_metrics(output_dir: Path, track_name: str, sources: List[str], mean_sdr, mean_sir, mean_sar, mean_isr) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = track_name.replace('/', '_').replace('\\', '_')
    out_path = output_dir / f"metrics_{safe_name}.json"

    data = {
        "track": track_name,
        "per_source": {},
        "overall": {
            "sdr": float(np.mean(mean_sdr)),
            "sir": float(np.mean(mean_sir)),
            "sar": float(np.mean(mean_sar)),
            "isr": float(np.mean(mean_isr)),
        },
    }
    for i, name in enumerate(sources):
        data["per_source"][name] = {
            "sdr": float(mean_sdr[i]),
            "sir": float(mean_sir[i]),
            "sar": float(mean_sar[i]),
            "isr": float(mean_isr[i]),
        }

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    print(f"Saved metrics to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute BSS metrics (museval) for a single MUSDB18-wav track")
    parser.add_argument("--data_root", type=str, default="./musdb18-wav", help="MUSDB18 data root directory")
    parser.add_argument("--track_name", type=str, default=DEFAULT_TRACK_NAME, help="Track name (directory in musdb18-wav/test)")
    parser.add_argument("--estimates_root", type=str, default=None, help="Directory with estimates or a parent directory that contains a track-named folder")
    parser.add_argument("--output_dir", type=str, default=None, help="If provided, metrics are saved as JSON in this directory")

    args = parser.parse_args()

    musdb_root = Path(args.data_root)
    track_name = args.track_name

    if args.estimates_root is None:
        estimates_root = Path("./separated") / "new" / track_name
    else:
        estimates_root = Path(args.estimates_root)

    output_dir = Path(args.output_dir) if args.output_dir is not None else None

    print("== BSS-Eval for a single MUSDB18-wav track (museval) ==")
    print(f"MUSDB root:   {musdb_root}")
    print(f"Estimates:    {estimates_root}")
    print(f"Track:        {track_name}\n")

    refs_np = load_reference_sources(musdb_root, track_name)
    ests_np = load_estimated_sources(estimates_root, track_name)

    L = min(refs_np.shape[1], ests_np.shape[1])
    refs_np = refs_np[:, :L]
    ests_np = ests_np[:, :L]

    print(f"refs shape: {refs_np.shape}, ests: {ests_np.shape}")

    print("\nComputing BSS metrics using museval.metrics.bss_eval_sources ...")
    sdr, sir, sar, isr = muse_metrics.bss_eval_sources(refs_np, ests_np)

    mean_sdr = sdr.mean(axis=1)
    mean_sir = sir.mean(axis=1)
    mean_sar = sar.mean(axis=1)
    mean_isr = isr.mean(axis=1)

    print("\nResults (mean across frames):")
    for i, name in enumerate(SOURCES):
        print(
            f"  {name:7s} | SDR: {mean_sdr[i]:6.2f} dB | "
            f"SIR: {mean_sir[i]:6.2f} dB | SAR: {mean_sar[i]:6.2f} dB | ISR: {mean_isr[i]:6.2f} dB"
        )

    print("\nMean across all sources:")
    print(f"  SDR: {mean_sdr.mean():6.2f} dB")
    print(f"  SIR: {mean_sir.mean():6.2f} dB")
    print(f"  SAR: {mean_sar.mean():6.2f} dB")
    print(f"  ISR: {mean_isr.mean():6.2f} dB")

    if output_dir is not None:
        try:
            save_metrics(output_dir, track_name, SOURCES, mean_sdr, mean_sir, mean_sar, mean_isr)
        except Exception as exc:
            print(f"Failed to save metrics: {exc}")


if __name__ == "__main__":
    main()
