"""``colorize`` command-line entry point."""

import argparse
from pathlib import Path

import torch

from image_colorizer import get_device
from image_colorizer.model import MainModel
from image_colorizer.visualize import ColorizationVisualizer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Colorize a black-and-white image.")
    parser.add_argument("--input", "-i", required=True, help="Path to the input image.")
    parser.add_argument(
        "--output",
        "-o",
        help="Where to write the colorized image (default: <input>_color.png).",
    )
    parser.add_argument(
        "--checkpoint", "-c", default="checkpoints/model.pth", help="Trained model weights."
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--show", action="store_true", help="Also display the result.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = get_device()

    model = MainModel(device=device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(
        f"{input_path.stem}_color.png"
    )

    viz = ColorizationVisualizer(model, size=args.image_size, device=device)
    viz.save_colorization(input_path, output_path)
    print(f"Saved {output_path}")
    if args.show:
        viz.display_colorization(input_path)


if __name__ == "__main__":
    main()
