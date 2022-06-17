import numpy as np
import torch
from torchvision import transforms


class FirstChannelRandomInvert(object):
    def __init__(self, p):
        self.p = p
        self.invert_color = transforms.RandomInvert(p=self.p)

    def __call__(self, sample):
        #input shape: (..., 4, H, W)
        image, mask = sample[0].unsqueeze(dim=0), sample[1:]
        image = self.invert_color(image)
        sample = torch.cat((image, mask), dim=0)
        return sample


class FirstChannelRandomGaussianBlur(object):
    def __init__(self, p, kernel_size=(5, 9), sigma=(0.1, 5)):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma)

    def __call__(self, sample):
        #input shape: (..., 4, H, W)
        image, mask = sample[0].unsqueeze(dim=0), sample[1:]
        randn = np.random.rand()
        if randn < self.p: image = self.gaussian_blur(image)
        sample = torch.cat((image, mask), dim=0)
        return sample


data_transforms = transforms.Compose([
  transforms.RandomHorizontalFlip(p=0.5),
  transforms.RandomVerticalFlip(p=0.5),
  FirstChannelRandomInvert(p=0.5),
  FirstChannelRandomGaussianBlur(p=0.5),
  transforms.RandomRotation((0, 360)),
  transforms.RandomPerspective(distortion_scale=0.6, p=0.5)
])