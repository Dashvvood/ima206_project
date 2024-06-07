import torchvision.transforms as transforms
from constant import PathMNIST_MEAN, PathMNIST_STD
import random
from PIL import Image, ImageOps, ImageFilter
import torch
from deprecated import deprecated

class Rotate90orMinus90(torch.nn.Module):
    
    def __init__(self, p) -> None:
        self.p = p
        
    def __call__(self, x):
        if random.random() <= self.p:
            angle = random.choice([90, -90])
            return transforms.functional.rotate(x, angle)
        return x

class StandardFinetuneTransform(object):
    """
    The same with the other models
    """
    
    def __init__(
        self, 
        img_size=28,
        flip_p = 0.5,
        rotate_p = 0.5,
        gaussian_p = 0.5,
    ) -> None:
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=img_size, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
            transforms.RandomHorizontalFlip(p=flip_p),
            Rotate90orMinus90(p=rotate_p),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=7)], p=gaussian_p),
            transforms.ToTensor(),
            transforms.Normalize(mean=PathMNIST_MEAN, std=PathMNIST_STD),
        ])
    
    def __call__(self, x):
        return self.transform(x)
    

class BarlowTwinPretainTransform(StandardFinetuneTransform):
    def __init__(self, img_size=28, flip_p=0.5, rotate_p=0.5, gaussian_p=0.5) -> None:
        super().__init__(img_size, flip_p, rotate_p, gaussian_p)
    
    def __call__(self, x):
        return self.transform(x), self.transform(x)
    

class FinetuneTransform(object):
    def __init__(
        self, 
        img_size=28,
        flip_p = 0.5,
        rotate_p = 0.5,
        gaussian_p = 0.5,
    ) -> None:
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=img_size, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
            transforms.RandomHorizontalFlip(p=flip_p),
            # Rotate90orMinus90(p=rotate_p),
            # transforms.RandomApply([transforms.GaussianBlur(kernel_size=7)], p=gaussian_p),
            transforms.ToTensor(),
            transforms.Normalize(mean=PathMNIST_MEAN, std=PathMNIST_STD),
        ])
    
    def __call__(self, x):
        return self.transform(x)


def pathmnist_normalization():
    return transforms.Normalize(mean=PathMNIST_MEAN, std=PathMNIST_STD)



@deprecated("This class is deprecated, please use ")
class BarlowTwinsTransform:
    def __init__(self, train=True, input_height=224, gaussian_blur=True, jitter_strength=1.0, normalize=None):
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.jitter_strength = jitter_strength
        self.normalize = normalize
        self.train = train

        color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        color_transform = [transforms.RandomApply([color_jitter], p=0.8), transforms.RandomGrayscale(p=0.2)]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1

            color_transform.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size)], p=0.5))

        self.color_transform = transforms.Compose(color_transform)

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.input_height),
                transforms.RandomHorizontalFlip(p=0.5),
                self.color_transform,
                self.final_transform,
            ]
        )

        self.finetune_transform = None
        if self.train:
            self.finetune_transform = transforms.Compose(
                [
                    transforms.RandomCrop(self.input_height, padding=4, padding_mode="reflect"),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.finetune_transform = transforms.ToTensor()

    def __call__(self, sample):
        return self.transform(sample), self.transform(sample), self.finetune_transform(sample)
    