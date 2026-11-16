import glob

import numpy as np
from fastai.data.external import URLs, untar_data  # type: ignore
from PIL import Image
from skimage.color import rgb2lab
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ColorizationDataLoader:
    def __init__(
        self,
        path="",
        num_images=10000,
        size=256,
        batch_size=32,
        n_workers=4,
        pin_memory=True,
    ):
        self.path = path
        self.num_images = num_images
        self.size = size
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.pin_memory = pin_memory
        self.train_paths, self.val_paths = self._prepare_paths()

    def _prepare_paths(self):
        if self.path == "":
            self.path = self._load_coco_dataset()

        paths = sorted(
            p for ext in ("jpg", "jpeg", "png") for p in glob.glob(f"{self.path}/*.{ext}")
        )
        if not paths:
            raise FileNotFoundError(f"No images found in {self.path!r}")
        self.num_images = min(self.num_images, len(paths))

        rng = np.random.default_rng(123)
        paths_subset = rng.choice(paths, self.num_images, replace=False)
        rand_idxs = rng.permutation(self.num_images)
        split = int(self.num_images * 0.8)
        train_idxs = rand_idxs[:split]  # first 80% for training
        val_idxs = rand_idxs[split:]  # remaining 20% for validation
        train_paths = paths_subset[train_idxs]
        val_paths = paths_subset[val_idxs]
        return train_paths, val_paths

    def _load_coco_dataset(self):
        coco_path = untar_data(URLs.COCO_SAMPLE)
        coco_path = str(coco_path) + "/train_sample"
        return coco_path

    class ColorizationDataset(Dataset):
        def __init__(self, paths, split="train", size=256):
            self.split = split
            self.size = size
            self.paths = paths

            if split == "train":
                self.transforms = transforms.Compose(
                    [
                        transforms.Resize(
                            (self.size, self.size), transforms.InterpolationMode.BICUBIC
                        ),
                        transforms.RandomHorizontalFlip(),  # A little data augmentation!
                    ]
                )
            elif split == "val":
                self.transforms = transforms.Resize(
                    (self.size, self.size), transforms.InterpolationMode.BICUBIC
                )

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            img = self.transforms(img)
            img = np.array(img)
            img_lab = rgb2lab(img).astype("float32")  # Converting RGB to L*a*b
            img_lab = transforms.ToTensor()(img_lab)
            L = img_lab[[0], ...] / 50.0 - 1.0  # Between -1 and 1
            ab = img_lab[[1, 2], ...] / 110.0  # Between -1 and 1

            return {"L": L, "ab": ab}

        def __len__(self):
            return len(self.paths)

    def get_dataloaders(self):
        train_dataset = self.ColorizationDataset(
            paths=self.train_paths, split="train", size=self.size
        )
        val_dataset = self.ColorizationDataset(paths=self.val_paths, split="val", size=self.size)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

        return train_loader, val_loader
