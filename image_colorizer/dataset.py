import glob
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from fastai.data.external import untar_data, URLs

class ColorizationDataLoader:
    def __init__(self, path='', num_images=10000, size=256, batch_size=32, n_workers=4, pin_memory=True):
        self.path = path
        self.num_images = num_images
        self.size = size
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.pin_memory = pin_memory
        self.train_paths, self.val_paths = self._prepare_paths()
        
    def _prepare_paths(self):
        if self.path == '':
            self.path = self._load_coco_dataset()
        
        paths = glob.glob(self.path + "/*.jpg")  # Grabbing all the image file names
        np.random.seed(123)
        paths_subset = np.random.choice(paths, self.num_images, replace=False)  # choosing images randomly
        rand_idxs = np.random.permutation(self.num_images)
        train_idxs = rand_idxs[:int(self.num_images * 0.8)]  # choosing the first 80% as training set
        val_idxs = rand_idxs[int(self.num_images * 0.2):]  # choosing last 20% as validation set
        train_paths = paths_subset[train_idxs]
        val_paths = paths_subset[val_idxs]
        return train_paths, val_paths

    def _load_coco_dataset(self):
        coco_path = untar_data(URLs.COCO_SAMPLE)
        coco_path = str(coco_path) + "/train_sample"
        return coco_path

    class ColorizationDataset(Dataset):
        def __init__(self, paths, split='train', size=256):
            self.split = split
            self.size = size
            self.paths = paths
            
            if split == 'train':
                self.transforms = transforms.Compose([
                    transforms.Resize((self.size, self.size), Image.BICUBIC),
                    transforms.RandomHorizontalFlip(),  # A little data augmentation!
                ])
            elif split == 'val':
                self.transforms = transforms.Resize((self.size, self.size), Image.BICUBIC)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            img = self.transforms(img)
            img = np.array(img)
            img_lab = rgb2lab(img).astype("float32")  # Converting RGB to L*a*b
            img_lab = transforms.ToTensor()(img_lab)
            L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1
            ab = img_lab[[1, 2], ...] / 110.  # Between -1 and 1

            return {'L': L, 'ab': ab}

        def __len__(self):
            return len(self.paths)

    def get_dataloaders(self):
        train_dataset = self.ColorizationDataset(paths=self.train_paths, split='train', size=self.size)
        val_dataset = self.ColorizationDataset(paths=self.val_paths, split='val', size=self.size)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.n_workers, pin_memory=self.pin_memory, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.n_workers, pin_memory=self.pin_memory, shuffle=False)
        
        return train_loader, val_loader
