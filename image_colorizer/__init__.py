import os

import torch

if torch.backends.mps.is_available():
    device = torch.device(
        "mps"
    )  # Metal Performance Shaders (MPS) for Mac with Apple Silicon
    device_name = "Mac with Apple Silicon"
elif torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
else:
    device = torch.device("cpu")
    device_name = "CPU"

print(f"Device: {device_name}")
print(f"Device type: {device}")
