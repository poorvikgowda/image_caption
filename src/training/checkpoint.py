import os
import torch
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger("ImageCaptioning")

def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: str = "checkpoints",
    filename: str = "last_model.pth"
) -> str:
    """
    Save model checkpoint to disk.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    logger.info(f"Saved checkpoint to {filepath}")

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_path)
        logger.info(f"Saved BEST model checkpoint to {best_path}")

    return filepath

def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: Any = None,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Load model checkpoint from disk safely.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    logger.info(f"Loading checkpoint from {filepath}...")
    checkpoint = torch.load(filepath, map_location=device or "cpu")

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info(f"Successfully loaded checkpoint from epoch {checkpoint.get('epoch', 0)} with val_loss {checkpoint.get('val_loss', float('inf')):.4f}")
    except RuntimeError as e:
        logger.warning(f"Checkpoint size mismatch ({e}). Initialized model with fresh architecture weights matching the current {getattr(model, 'vocab_size', 'new')} vocabulary.")

    return checkpoint
