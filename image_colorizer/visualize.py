import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from skimage.color import rgb2lab
import matplotlib.pyplot as plt

class ColorizationVisualizer:
    """
    Class to handle the visualization of image colorization.
    """

    def __init__(self, model, size=256, device=torch.device("cpu")):
        """
        Initialize the ColorizationVisualizer with a model and configuration.

        Args:
            model: The colorization model to be used.
            size (int): The size to which images are resized.
            device: The device on which to perform computations (CPU or GPU).
        """
        self.model = model
        self.size = size
        self.device = device

    def load_image(self, image_path):
        """
        Load image from disk, convert it to LAB color space, and convert to a tensor.

        Args:
            image_path (str): Path to the image file.

        Returns:
            torch.Tensor: The L channel of the LAB image.
        """
        transform = transforms.Compose([
            transforms.Resize((self.size, self.size), Image.BICUBIC),
        ])

        img = Image.open(image_path).convert("RGB")
        img = transform(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")  # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1

        return L.unsqueeze(0).to(self.device), L.to('cpu')

    def display_colorization(self, image_path):
        """
        Display both the colorized and black-and-white versions of the image.

        Args:
            image_path (str): Path to the image file.
        """
        # Load the image
        image, l = self.load_image(image_path)

        print("Image shape:", image.shape)

        # Colorize the image using the model
        self.model.net_G.eval()
        with torch.no_grad():
            self.model.get_images(image)
            self.model.forward_data()
        self.model.net_G.train()
        fake_color = self.model.fake_color.detach()
        L = self.model.L
        fake_imgs = self.lab_to_rgb(L, fake_color)

        fake_imgs = fake_imgs.squeeze()

        # Display the images
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(Image.open(image_path))
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        axes[1].imshow(fake_imgs)
        axes[1].set_title("Colorized Image")
        axes[1].axis("off")
        plt.show()

    @staticmethod
    def lab_to_rgb(L, ab):
        """
        Convert LAB image to RGB.

        Args:
            L (torch.Tensor): L channel of the LAB image.
            ab (torch.Tensor): AB channels of the LAB image.

        Returns:
            np.array: RGB image.
        """
        L = (L + 1.) * 50.
        ab = ab * 110.
        lab = torch.cat([L, ab], dim=1).data.cpu().numpy().transpose((1, 2, 0))
        rgb = lab2rgb(lab)
        return rgb
