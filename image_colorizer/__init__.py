"""Image colorization with a U-Net generator trained as a conditional GAN."""

import torch

__version__ = "0.1.0"


def get_device() -> torch.device:
    """Return the best available device: Apple MPS, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
