import torch

from image_colorizer.helper import lab_to_rgb
from image_colorizer.model import MainModel
from image_colorizer.model.patch_discriminator import PatchDiscriminator
from image_colorizer.model.unet import Unet


def test_unet_output_shape():
    net = Unet(input_c=1, output_c=2, n_down=8, num_filters=8).eval()
    out = net(torch.randn(2, 1, 256, 256))
    assert out.shape == (2, 2, 256, 256)
    assert out.abs().max() <= 1.0  # tanh output


def test_patch_discriminator_output_shape():
    net = PatchDiscriminator(input_c=3, n_down=3, num_filters=8)
    out = net(torch.randn(2, 3, 256, 256))
    assert out.shape == (2, 1, 30, 30)


def test_lab_to_rgb_range():
    rgb = lab_to_rgb(torch.zeros(1, 1, 8, 8), torch.zeros(1, 2, 8, 8))
    assert rgb.shape == (1, 8, 8, 3)
    assert 0.0 <= rgb.min() and rgb.max() <= 1.0


def test_one_optimization_step_runs():
    model = MainModel(device=torch.device("cpu"))
    batch = {"L": torch.randn(1, 1, 256, 256), "ab": torch.rand(1, 2, 256, 256) * 2 - 1}
    model.setup_input(batch)
    model.optimize()
    assert torch.isfinite(model.loss_G) and torch.isfinite(model.loss_D)
