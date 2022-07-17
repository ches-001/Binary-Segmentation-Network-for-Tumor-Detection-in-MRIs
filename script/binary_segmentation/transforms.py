import numpy as np
import torch, random
from torchvision import transforms
from typing import Union, Optional


class Noise:
    
    @staticmethod
    def apply_gaussian(
        image:torch.Tensor, mean:Optional[Union[int, float]]=0, std:Optional[Union[int, float]]=1):
        noise = torch.normal(mean=mean, std=std, size=image.shape)
        image = image + noise
        return image

    @staticmethod
    def apply_speckle(
        image:torch.Tensor, mean:Optional[Union[int, float]]=0, std:Optional[Union[int, float]]=1):
        speckle = image * Noise.apply_gaussian(image, mean, std)
        image = image + speckle
        return image

    @staticmethod
    def apply_salt_and_pepper(image:torch.Tensor, n_pixels:int=500):
        _, H, W = image.shape

        for op in ['salt', 'pepper']:
            val = image.max()
            if op == 'pepper':val = 0
            
            for i in range(n_pixels):
                x_coord = random.randint(0, W-1)
                y_coord = random.randint(0, H-1)
                image[:, y_coord, x_coord] = val
        return image
        
    @staticmethod
    def apply_poisson(image:torch.Tensor, rate:Optional[Union[int, float]]=5):
        noise = torch.rand(*image.shape) * rate
        noise = torch.poisson(noise)
        image = image + noise
        return image


class FirstChannelRandomGaussianBlur(object):
    def __init__(self, p:float, kernel_size:tuple=(3, 9), sigma:tuple=(0.1, 11)):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma)

    def __call__(self, sample:torch.Tensor):
        #input shape: (4, H, W)
        if self.p < random.random():return sample

        image, mask = sample[0].unsqueeze(dim=0), sample[1:]
        image = self.gaussian_blur(image)
        sample = torch.cat((image, mask), dim=0)
        return sample


class FirstChannelRandomNoise(object):
    def __init__(self, p:float=0.5, **kwargs):
        defaultKwargs = {
            'mean_range':(0, 50),
            'std_range':(1, 50),
            'sp_px_range':(300, 5000),
            'poisson_rate_range':(5, 100)}
        
        self.p = p
        self.kwargs = {**defaultKwargs, **kwargs}
        self.ops = ('gaussian', 'speckle', 'salt_&_pepper', 'poisson')

    def __call__(self, sample:torch.Tensor):
        if self.p < random.random():return sample

        image, mask = sample[0].unsqueeze(dim=0), sample[1:]
        op = random.choice(self.ops)
        if op == 'gaussian':
            mean = random.randint(*self.kwargs['mean_range'])
            std = random.randint(*self.kwargs['std_range'])
            image = Noise.apply_gaussian(image, mean, std)

        elif op == 'speckle':
            mean = random.randint(*self.kwargs['mean_range'])
            std = random.randint(*self.kwargs['std_range'])
            image = Noise.apply_speckle(image, mean, std)

        elif op == 'salt_&_pepper':
            n_pixels = random.randint(*self.kwargs['sp_px_range'])
            image = Noise.apply_salt_and_pepper(image, n_pixels)

        elif op == 'poisson':
            rate = random.randint(*self.kwargs['poisson_rate_range'])
            image = Noise.apply_poisson(image, rate)

        sample = torch.cat((image, mask), dim=0)
        return sample

        
class CustomRandomRotation(object):
    def __init__(self, p:float, angle_range:tuple=(0, 360)):
        self.p = p
        self.angle_range = angle_range
        self.random_rotation = transforms.RandomRotation(self.angle_range)

    def __call__(self, sample:torch.Tensor):
        #input shape: (C, H, W)
        if self.p < random.random():return sample

        sample = self.random_rotation(sample)
        return sample

  
class CustomRandomResizedCrop(object):
    def __init__(self, p:float, scale:tuple=(0.5, 1.0)):
        self.p = p
        self.scale = scale

    def __call__(self, sample:torch.Tensor):
        #input shape: (C, H, W)
        if self.p < random.random():return sample

        _, H, W = sample.shape
        random_resizedcrop = transforms.RandomResizedCrop(
            (H, W), scale=self.scale, interpolation=transforms.InterpolationMode.NEAREST)
        sample = random_resizedcrop(sample)
        return sample


class CustomCompose(transforms.Compose):
    def __init__(self, transforms:list, shuffle:bool=False):
        super(CustomCompose, self).__init__(transforms)
        self.transforms = transforms
        self.shuffle = shuffle

    def __call__(self, image:torch.Tensor):
        transforms = self.transforms
        if self.shuffle:
            transforms = random.sample(self.transforms, len(self.transforms))
            
        for T in transforms:
            image = T(image)
        return image


def image_resize(image:Union[torch.Tensor, np.ndarray], size:Union[int, tuple]):
    #image shape: C, H, W or N, C, H, W
    if isinstance(size, int): H, W = size, size
    elif isinstance(size, tuple): H, W = size
    else: raise TypeError(f'size is required to be of type int fo tuple, {type(size)} given')

    assert len(image.shape) == 3 or len(image.shape) == 4, \
     'input image must be of shape  C, H, W or N, C, H, W'

    if isinstance(image, np.ndarray): image = torch.from_numpy(image) 
    res_model = transforms.Resize((H, W), interpolation=transforms.InterpolationMode.NEAREST)
    image = res_model(image)
    return image


def image_normalize(image:Union[torch.Tensor, np.ndarray]):
    #image size: (C, H, W)
    image = image.max() - image
    image = image - image.min()
    image = image / image.max()
    return image    


def data_augmentation(**kwargs):

    defaultKwargs = {
        'shuffle_tranforms':True,
        'crop_p':0.5, 
        'rotation_p':0.5, 
        'Hflip_p':0.5, 
        'Vflip_p':0.5, 
        'blur_p':0.5,
        'noise_p':0.5,
        'rotation_angle_range':(-60, 60),
        'crop_scale':(0.5, 1.0),
        'blur_kernel_size':(3, 9),
        'blur_sigma':(0.1, 11),
        }

    kwargs = {**defaultKwargs, **kwargs}

    transform_list = [
        CustomRandomResizedCrop(p=kwargs['crop_p'], scale=kwargs['crop_scale']),
        CustomRandomRotation(p=kwargs['rotation_p'], angle_range=kwargs['rotation_angle_range']),
        transforms.RandomHorizontalFlip(kwargs['Hflip_p']),
        transforms.RandomVerticalFlip(kwargs['Vflip_p']),
        FirstChannelRandomGaussianBlur(kwargs['blur_p'], kernel_size=kwargs['blur_kernel_size'], sigma=kwargs['blur_sigma']),
        FirstChannelRandomNoise(p=kwargs['noise_p'])
    ]
    T = CustomCompose(transform_list, shuffle=kwargs['shuffle_tranforms'])
    return T

if __name__ == '__main__':
    data_transforms = data_augmentation()