import warnings
import torch
from torch.utils.data import DataLoader

from training.data import MusdbRandomChunks
from training.model import apply_masks, reconstruct_sources
from training import losses

try:
    import mir_eval.separation as mir_eval_sep
except Exception:
    mir_eval_sep = None


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
    sdr_val = losses.sdr(preds, targets, reduction="mean")
    si_sdr_val = losses.si_sdr(preds, targets, reduction="mean")

    return {
        "sdr": float(sdr_val.item() if torch.is_tensor(sdr_val) else sdr_val),
        "si_sdr": float(si_sdr_val.item() if torch.is_tensor(si_sdr_val) else si_sdr_val),
    }


@torch.no_grad()
def compute_bss_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """Compute BSS Eval metrics (SDR, SIR, SAR) using mir_eval.

    Args:
        preds, targets: tensors of shape (B, S, 1, L) or (B, S, L).

    Returns:
        dict with keys 'sdr', 'sir', 'sar' containing the mean over batch and sources.

    Notes:
        Requires `mir_eval` to be installed. If not available, raises ImportError with guidance.
    """
    if mir_eval_sep is None:
        raise ImportError("mir_eval is required for BSS Eval metrics. Install it with: pip install mir_eval")

    # Align shapes and time length
    if preds.dim() == 4:
        preds = preds.squeeze(2)
    if targets.dim() == 4:
        targets = targets.squeeze(2)

    B, S, Lp = preds.shape
    _, _, Lt = targets.shape
    L = min(Lp, Lt)
    preds = preds[..., :L]
    targets = targets[..., :L]

    total_sdr = 0.0
    total_sir = 0.0
    total_sar = 0.0
    n_items = 0

    # mir_eval expects shape (nsrc, nsamples) as numpy arrays
    for b in range(B):
        # (S, L)
        est = preds[b].detach().cpu().numpy()
        ref = targets[b].detach().cpu().numpy()

        # Ensure arrays are (nsrc, nsamples)
        # mir_eval.bss_eval_sources returns (sdr, sir, sar, perm)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            sdr, sir, sar, _ = mir_eval_sep.bss_eval_sources(ref, est)

        total_sdr += float(sdr.mean())
        total_sir += float(sir.mean())
        total_sar += float(sar.mean())
        n_items += 1

    if n_items == 0:
        return {"sdr": float("nan"), "sir": float("nan"), "sar": float("nan")}

    return {"sdr": total_sdr / n_items, "sir": total_sir / n_items, "sar": total_sar / n_items}


def evaluate(
    model: torch.nn.Module,
    data_root: str,
    data_format: str,
    sources: tuple[str, ...],
    segment_seconds: float,
    device: torch.device,
    max_batches: int = 20,
    num_workers: int = 0,
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
        batch_size=1,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    total_sdr = 0.0
    total_si_sdr = 0.0
    n = 0
    bss_stats = {"sdr": 0.0, "sir": 0.0, "sar": 0.0}
    mir_eval_available = mir_eval_sep is not None

    model.eval()

    for i, batch in enumerate(loader_val):
        if i >= max_batches:
            break

        # batch["mixture"] shape from dataset: (C, L) because mono=True
        # DataLoader batch with batch_size=1 -> (B=1, C, L)
        mixture = batch["mixture"].to(device, non_blocking=True)  # (B, 1, L)
        targets = batch["targets"].to(device, non_blocking=True)  # (B, S, L)

        masks = model(mixture)  # (B, S, Lm)
        preds = reconstruct_sources(mixture, masks).unsqueeze(2)  # (B, S, 1, L)
        targets_batch = targets.unsqueeze(2)

        metrics = compute_sdr_metrics(preds, targets_batch)
        total_sdr += metrics["sdr"]
        total_si_sdr += metrics["si_sdr"]
        if mir_eval_available:
            try:
                bss = compute_bss_metrics(preds, targets_batch)
                for k in bss_stats:
                    bss_stats[k] += bss[k]
            except Exception as exc:
                print(f"Warning: compute_bss_metrics failed on batch {i}: {exc}")
                mir_eval_available = False
        n += 1

    model.train()

    if n == 0:
        return {"sdr": float("nan"), "si_sdr": float("nan")}

    result = {"sdr": total_sdr / n, "si_sdr": total_si_sdr / n}
    if bss_stats["sdr"] != 0 or bss_stats["sir"] != 0 or bss_stats["sar"] != 0:
        result.update({k: v / n for k, v in bss_stats.items()})
    return result
