import torch
import logging

logger = logging.getLogger("ImageCaptioning")

def get_device(force_cpu: bool = False) -> torch.device:
    """
    Get PyTorch device, prioritizing CUDA GPU if available.
    """
    if force_cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return device
