"""Inference helpers: colorize a single image with a trained model."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.color import rgb2lab
from torchvision import transforms

from image_colorizer.helper import lab_to_rgb


def _to_L_tensor(img: Image.Image) -> torch.Tensor:
    """Convert an RGB PIL image to a normalized L tensor of shape (1, 1, H, W)."""
    lab = rgb2lab(np.array(img)).astype("float32")
    L = transforms.ToTensor()(lab)[[0], ...] / 50.0 - 1.0
    return L.unsqueeze(0)


class ColorizationVisualizer:
    """Colorize images with a trained :class:`MainModel`."""

    def __init__(self, model, size=256, device=None):
        self.model = model
        self.size = size
        self.device = device if device is not None else model.device

    @torch.no_grad()
    def colorize(self, image_path) -> np.ndarray:
        """Return the colorized image as an RGB float array at its original resolution.

        The generator runs at ``size`` x ``size``; the predicted ab channels are
        upsampled and combined with the full-resolution L channel.
        """
        original = Image.open(image_path).convert("RGB")
        small = original.resize((self.size, self.size), Image.Resampling.BICUBIC)

        self.model.net_G.eval()
        ab = self.model.net_G(_to_L_tensor(small).to(self.device)).cpu()

        L_full = _to_L_tensor(original)
        ab_full = F.interpolate(ab, size=L_full.shape[-2:], mode="bilinear", align_corners=False)
        return lab_to_rgb(L_full, ab_full)[0]

    def save_colorization(self, image_path, output_path):
        rgb = np.clip(self.colorize(image_path), 0.0, 1.0)
        Image.fromarray((rgb * 255).round().astype("uint8")).save(output_path)

    def display_colorization(self, image_path):
        """Show the original and colorized images side by side."""
        rgb = self.colorize(image_path)
        _, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(Image.open(image_path).convert("L"), cmap="gray")
        axes[0].set_title("Input")
        axes[1].imshow(rgb)
        axes[1].set_title("Colorized")
        for ax in axes:
            ax.axis("off")
        plt.show()
