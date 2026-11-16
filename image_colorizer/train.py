"""Training loop and ``colorize-train`` command-line entry point."""

import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm

from image_colorizer import get_device
from image_colorizer.dataset import ColorizationDataLoader
from image_colorizer.helper import (
    create_loss_meters,
    log_results,
    update_losses,
    visualize,
)
from image_colorizer.model import MainModel


class ColorizationTrainer:
    """Run the GAN training loop for a :class:`MainModel`."""

    def __init__(
        self,
        model,
        train_dl,
        val_dl,
        epochs,
        display_every=200,
        save_path="checkpoints/model.pth",
        output_dir="outputs",
    ):
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.epochs = epochs
        self.display_every = display_every
        self.save_path = Path(save_path)
        self.output_dir = Path(output_dir)

    def save_checkpoint(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.save_path)

    def train(self):
        # Fixed validation batch for visualising progress at regular intervals.
        val_batch = next(iter(self.val_dl))
        for epoch in range(1, self.epochs + 1):
            loss_meters = create_loss_meters()
            for i, data in enumerate(tqdm(self.train_dl, desc=f"Epoch {epoch}"), 1):
                self.model.setup_input(data)
                self.model.optimize()
                update_losses(self.model, loss_meters, count=data["L"].size(0))
                if i % self.display_every == 0:
                    print(f"\nEpoch {epoch}/{self.epochs}, iteration {i}/{len(self.train_dl)}")
                    log_results(loss_meters)
                    visualize(self.model, val_batch, save_dir=self.output_dir)

            print(f"\nEpoch {epoch}/{self.epochs} finished")
            log_results(loss_meters)
            self.save_checkpoint()
            print(f"Checkpoint saved to {self.save_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the image colorization GAN.")
    parser.add_argument(
        "--data",
        default="",
        help="Folder of training images. Defaults to downloading the fastai COCO sample.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-images", type=int, default=10000)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--display-every", type=int, default=200)
    parser.add_argument("--checkpoint", default="checkpoints/model.pth")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = get_device()
    print(f"Using device: {device}")

    loaders = ColorizationDataLoader(
        path=args.data,
        num_images=args.num_images,
        size=args.image_size,
        batch_size=args.batch_size,
        n_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    train_dl, val_dl = loaders.get_dataloaders()

    model = MainModel(device=device)
    trainer = ColorizationTrainer(
        model,
        train_dl,
        val_dl,
        epochs=args.epochs,
        display_every=args.display_every,
        save_path=args.checkpoint,
        output_dir=args.output_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
