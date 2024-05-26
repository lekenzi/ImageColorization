#

import os
import glob
import time
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
import cv2
import os

if torch.backends.mps.is_available():
    device = torch.device("mps")  # Metal Performance Shaders (MPS) for Mac with Apple Silicon
    device_name = "Mac with Apple Silicon"
elif torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
elif 'COLAB_TPU_ADDR' in os.environ:
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        device_name = "TPU"
    except ImportError:
        device = torch.device("cpu")
        device_name = "CPU (TPU support not available)"
else:
    device = torch.device("cpu")
    device_name = "CPU"

print(f"Device: {device_name}")
print(f"Device type: {device}")

