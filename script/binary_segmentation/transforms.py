import numpy as np
import torch
from torchvision import transforms


class FirstChannelRandomGaussianBlur(object):
    def __init__(self, p, kernel_size=(5, 9), sigma=(0.1, 5)):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma)

    def __call__(self, sample):
        #input shape: (4, 224, 224)
        image, mask = sample[0].unsqueeze(dim=0), sample[1:]
        randn = np.random.rand()
        if randn < self.p: image = self.gaussian_blur(image)
        sample = torch.cat((image, mask), dim=0)
        return sample


class CustomRandomRotation(object):
    def __init__(self, p, angle_range=(0, 360)):
        self.p = p
        self.angle_range = angle_range
        self.random_rotation = transforms.RandomRotation(self.angle_range)

    def __call__(self, sample):
        #input shape: (C, 224, 224)
        randn = np.random.rand()
        if randn < self.p:
            sample = self.random_rotation(sample)
            sample[:, 1:, ...] = sample[:, 1:, ...].round()
        return sample

  
class CustomRandomResizedCrop(object):
    def __init__(self, p, size=224, scale=(0.3, 1.0)):
        self.p = p
        self.size = size
        self.scale = scale
        self.random_rotation = transforms.RandomResizedCrop(
            self.size, scale=self.scale, interpolation=transforms.InterpolationMode.NEAREST)

    def __call__(self, sample):
        #input shape: (C, 224, 224)
        randn = np.random.rand()
        if randn < self.p: 
            sample = self.random_rotation(sample)
            sample[:, 1:, ...] = sample[:, 1:, ...].round()
        return sample
        


data_transforms = transforms.Compose([
  CustomRandomResizedCrop(p=0.5),
  CustomRandomRotation(p=0.5, angle_range=(0, 60)),
  transforms.RandomHorizontalFlip(p=0.5),
  transforms.RandomVerticalFlip(p=0.5),
  FirstChannelRandomGaussianBlur(p=0.5),
])