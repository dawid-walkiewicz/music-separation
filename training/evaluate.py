import torch
from torch.utils.data import DataLoader

from training.data import MusdbRandomChunks
from training.model import apply_masks


@torch.no_grad()
def compute_sdr_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """Compute simple SDR and SI-SDR metrics (in dB) for a batch.

    This is used only for benchmarking/evaluation, not as a loss.

    Args:
        preds: Predicted sources, shape (B, S, 1, L).
        targets: Reference sources, shape (B, S, 1, L).

    Returns:
        dict with keys:
            - "sdr": mean SDR over batch and sources [dB]
            - "si_sdr": mean scale-invariant SDR over batch and sources [dB]
    """
    # Make sure time dimensions match
    L = min(preds.shape[-1], targets.shape[-1])
    preds = preds[..., :L]
    targets = targets[..., :L]

    # (B, S, 1, L) -> (B*S, L)
    B, S, C, L = preds.shape
    preds_f = preds.reshape(B * S, L)
    targets_f = targets.reshape(B * S, L)

    # Standard SDR: 10 * log10( ||s||^2 / ||s - s_hat||^2 )
    eps = 1e-8
    num = (targets_f ** 2).sum(dim=-1)
    den = ((targets_f - preds_f) ** 2).sum(dim=-1) + eps
    sdr = 10.0 * torch.log10(num / den + eps)

    # SI-SDR (scale-invariant SDR)
    # proj = <s_hat, s> / ||s||^2 * s
    dot = (preds_f * targets_f).sum(dim=-1)
    s_target_energy = (targets_f ** 2).sum(dim=-1) + eps
    scale = dot / s_target_energy
    proj = scale.unsqueeze(-1) * targets_f
    e_noise = preds_f - proj
    si_num = (proj ** 2).sum(dim=-1)
    si_den = (e_noise ** 2).sum(dim=-1) + eps
    si_sdr = 10.0 * torch.log10(si_num / si_den + eps)

    return {
        "sdr": sdr.mean().item(),
        "si_sdr": si_sdr.mean().item(),
    }


def evaluate(
    model: torch.nn.Module,
    data_root: str,
    data_format: str,
    sources: tuple[str, ...],
    segment_seconds: float,
    device: torch.device,
    max_batches: int = 20,
) -> dict:
    """Run a quick benchmark on random chunks from the MUSDB18 test subset.

    This gives an approximate SDR / SI-SDR score for the current model
    without evaluating on the full test set.
    """
    dataset_val = MusdbRandomChunks(
        root=data_root,
        data_format=data_format,
        subset="test",
        sources=list(sources),
        segment_seconds=segment_seconds,
        items_per_epoch=max_batches,
        mono=True,
        seed=123,
    )

    loader_val = DataLoader(
        dataset_val,
        batch_size=1,  # same as training: mixture (B=1, C=1, L)
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    total_sdr = 0.0
    total_si_sdr = 0.0
    n = 0

    model.eval()

    for i, batch in enumerate(loader_val):
        if i >= max_batches:
            break

        # batch["mixture"] shape from dataset: (C, L) because mono=True
        # DataLoader batch with batch_size=1 -> (B=1, C, L)
        mixture = batch["mixture"].to(device, non_blocking=True)  # (1, 1, L)
        targets = batch["targets"].to(device, non_blocking=True)  # (1, S, 1, L)

        masks = model(mixture)  # expects (B, C, L)
        preds = apply_masks(mixture, masks)  # (B, S, 1, L)

        metrics = compute_sdr_metrics(preds, targets)
        total_sdr += metrics["sdr"]
        total_si_sdr += metrics["si_sdr"]
        n += 1

    model.train()

    if n == 0:
        return {"sdr": float("nan"), "si_sdr": float("nan")}

    return {"sdr": total_sdr / n, "si_sdr": total_si_sdr / n}
