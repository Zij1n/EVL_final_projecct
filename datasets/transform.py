import torchvision.transforms as T
from PIL import ImageFilter
import numpy as np
import einops
import torch


class GaussianBlur:
    """Applies Gaussian blur to a batch of images with consistent sigma across the batch."""

    def __init__(self, sigma):
        self.sigma = sigma
        self.to_pil = T.ToPILImage()
        self.to_tensor = T.ToTensor()

    def __call__(self, x):
        sigma = self.sigma[0] + (self.sigma[1] - self.sigma[0]) * np.random.rand()
        blurred_images = torch.stack(
            [
                self.to_tensor(self.to_pil(img).filter(ImageFilter.GaussianBlur(sigma)))
                for img in x
            ]
        )

        return blurred_images


def handle_dims(transform):
    """Applies the given transform only to each image in the first element of the input list."""

    def wrapped_transform(inputs):
        images, *rest = inputs
        orig_shape = images.shape
        images = einops.rearrange(images, "... C H W -> (...) C H W")
        transformed_images = transform(images)
        transformed_images = transformed_images.reshape(*orig_shape)

        return [transformed_images, *rest]

    return wrapped_transform


def identity():
    """A transform that returns the input image batch as-is."""
    return lambda x: x


def random_resized_crop():
    """Applies a random crop with the specified crop size to each image in the batch."""
    return handle_dims(
        T.Compose(
            [
                T.RandomResizedCrop(
                    224, scale=(0.6, 1.0), ratio=(0.75, 1.33), antialias=True
                ),
            ]
        )
    )


def moco_v1():
    """MoCo v1 transform: augmentation pipeline as in InstDisc, applied to each image in the batch."""
    return handle_dims(
        T.Compose(
            [
                T.RandomResizedCrop(224, scale=(0.6, 1.0), antialias=True),
                T.RandomGrayscale(p=0.2),
                T.ColorJitter(0.4, 0.4, 0.4, 0.4),
                T.RandomHorizontalFlip(),
            ]
        )
    )


def moco_v2():
    """MoCo v2 transform: similar to SimCLR, applied to each image in the batch."""
    sigma_range = (0.1, 2.0)
    max_sigma = sigma_range[1]
    kernel_size = int(max_sigma * 3) | 1  # Ensure kernel size is odd

    return handle_dims(
        T.Compose(
            [
                T.RandomResizedCrop(224, scale=(0.6, 1.0), antialias=True),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply(
                    [T.GaussianBlur(kernel_size=kernel_size, sigma=sigma_range)], p=0.5
                ),
                T.RandomHorizontalFlip(),
            ]
        )
    )


def moco_v2_pil():
    """MoCo v2 transform with custom Gaussian blur, handling extra dimensions in the batch."""
    sigma_range = (0.1, 2.0)

    return handle_dims(
        T.Compose(
            [
                T.RandomResizedCrop(224, scale=(0.6, 1.0), antialias=True),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([GaussianBlur(sigma_range)], p=0.5),
                T.RandomHorizontalFlip(),
            ]
        )
    )


def custom_transform(**flags):
    """
    Creates a composed transform based on individual True or False flags passed as keyword arguments.
    Each flag corresponds to a specific transform:
    - random_resized_crop: RandomResizedCrop
    - color_jitter: RandomApply(ColorJitter)
    - random_grayscale: RandomGrayscale
    - gaussian_blur: RandomApply(GaussianBlur)
    - horizontal_flip: RandomHorizontalFlip
    """
    transform_map = {
        "random_resized_crop": T.RandomResizedCrop(
            224, scale=(0.6, 1.0), antialias=True
        ),
        "color_jitter": T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        "random_grayscale": T.RandomGrayscale(p=0.2),
        "gaussian_blur": T.RandomApply([GaussianBlur((0.1, 2.0))], p=0.5),
        "horizontal_flip": T.RandomHorizontalFlip(),
    }

    # Select transforms based on the flags
    selected_transforms = [
        transform for name, transform in transform_map.items() if flags.get(name, False)
    ]

    composed_transform = T.Compose(selected_transforms)
    return handle_dims(composed_transform)
