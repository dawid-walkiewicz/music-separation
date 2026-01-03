from typing import Dict, List, Mapping

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from models.audio_functions import ensure_stereo
from models.unet2d.common_functions import ensure_target_wave_dims
from models.unet2d.model import Unet2DWrapper
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from training.data import MusdbRandomChunks


def _evaluate_batch(
    model: Unet2DWrapper,
    batch: Dict[str, object],
    sources: List[str],
    device: torch.device,
    use_cuda: bool,
) -> Dict[str, float]:
    """Evaluate SI-SDR metrics on a single batch.

    Returns per-stem and mean SI-SDR in a dict with keys like
    "si_sdr/vocals", ..., "si_sdr/mean".
    """
    mixture = batch["mixture"].to(device, non_blocking=True)  # (B, C, L)
    targets_dict = batch["targets"]  # Dict[str, Tensor], already (B, C, L)

    mixture_stereo, B, L = ensure_stereo(mixture)

    si_sdr_per_stem: Dict[str, Tensor] = {name: torch.zeros((), device=device) for name in sources}
    count_per_stem: Dict[str, int] = {name: 0 for name in sources}

    sisdr_metric = ScaleInvariantSignalDistortionRatio().to(device)

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=use_cuda):
        for b in range(B):
            wav_stereo = mixture_stereo[b]  # (2, L)

            masked_stfts = model(wav_stereo)  # dict[name -> (2, F, T, 1) complex]

            for stem_name in sources:
                if stem_name not in masked_stfts:
                    continue
                if stem_name not in targets_dict:
                    continue

                pred_stft = masked_stfts[stem_name]
                target_wave = targets_dict[stem_name][b].to(device, non_blocking=True)  # (C, L)

                target_wave = ensure_target_wave_dims(target_wave, L)

                pred_wave = model.inverse_stft(pred_stft)  # (2, L_pred)

                min_len = min(pred_wave.shape[-1], target_wave.shape[-1])
                if pred_wave.shape[-1] != target_wave.shape[-1]:
                    pred_wave = pred_wave[..., :min_len]
                    target_wave = target_wave[..., :min_len]

                pred_in = pred_wave.unsqueeze(0) # (1, C, L)
                target_in = target_wave.unsqueeze(0) # (1, C, L)

                si_sdr_value = sisdr_metric(pred_in, target_in)

                si_sdr_per_stem[stem_name] += si_sdr_value

                count_per_stem[stem_name] += 1

    metrics: Dict[str, float] = {}

    total_si_sdr = 0.0
    used_stems = 0
    for stem_name in sources:
        n = count_per_stem[stem_name]
        if n == 0:
            continue
        value = (si_sdr_per_stem[stem_name] / n).item()
        metrics[f"si_sdr/{stem_name}"] = float(value)
        total_si_sdr += value
        used_stems += 1
    if used_stems > 0:
        metrics["si_sdr/mean"] = float(total_si_sdr / used_stems)

    return metrics


def evaluate(
    model: Unet2DWrapper,
    data_root: str,
    sources: List[str],
    segment_seconds: float,
    device: torch.device,
    subset: str = "test",
    batch_size: int = 4,
    num_workers: int = 4,
) -> Dict[str, float]:
    """Evaluate Unet2DWrapper on MUSDB18 split using SI-SDR.

    Args:
        model:          Trained Unet2DWrapper.
        data_root:      Root directory of MUSDB18 (wav or stem).
        sources:        List of stem names, must match model.stems.
        segment_seconds: Length of evaluation segments.
        device:         torch.device.
        subset:         MUSDB subset: "test" (default) or "train"/"valid".
        batch_size:     Batch size for evaluation loader.
        num_workers:    Number of DataLoader workers.

    Returns:
        Dict[str, float]: aggregate SI-SDR metrics per stem and mean SI-SDR.
    """
    dataset = MusdbRandomChunks(
        root=data_root,
        subset=subset,
        sources=list(sources),
        segment_seconds=segment_seconds,
        items_per_epoch=20,
        mono=False,
        seed=42,
    )

    use_cuda = device.type == "cuda"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    agg: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for i, batch in enumerate(loader, start=1):
        batch_metrics = _evaluate_batch(model, batch, sources, device, use_cuda)

        for key, value in batch_metrics.items():
            if key not in agg:
                agg[key] = 0.0
                counts[key] = 0
            agg[key] += float(value)
            counts[key] += 1

    final_metrics: Dict[str, float] = {}
    for key, total in agg.items():
        cnt = counts[key]
        if cnt > 0:
            final_metrics[key] = total / cnt

    return final_metrics
