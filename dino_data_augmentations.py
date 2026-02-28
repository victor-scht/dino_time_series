from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps
import random


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class DataAugmentationDINO:
    def __init__(
        self,
        global_crops_scale=(0.25, 1.0),
        local_crops_scale=(0.05, 0.25),
        local_crops_number=8,
    ):
        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),  # CIFAR-10 stats
                    (0.2470, 0.2435, 0.2616),
                ),
            ]
        )

        # First global crop: always blurred
        self.global_transfo1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    32, scale=global_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                GaussianBlur(p=1.0, radius_min=0.1, radius_max=1.0),
                normalize,
            ]
        )

        # Second global crop: sometimes blurred + solarized
        self.global_transfo2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    32, scale=global_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                GaussianBlur(p=0.1, radius_min=0.1, radius_max=1.0),
                Solarization(p=0.2),
                normalize,
            ]
        )

        # Local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    16, scale=local_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                GaussianBlur(p=0.5, radius_min=0.1, radius_max=0.5),
                normalize,
            ]
        )

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
