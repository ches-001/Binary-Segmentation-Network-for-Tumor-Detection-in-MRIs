import numpy as np
import torch, random
from torchvision import transforms


class FirstChannelRandomGaussianBlur(object):
    def __init__(self, p, kernel_size=(5, 9), sigma=(0.1, 11)):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma)

    def __call__(self, sample):
        #input shape: (4, H, W)
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
        #input shape: (C, H, W)
        randn = np.random.rand()
        if randn < self.p:
            sample = self.random_rotation(sample)
            sample[1:, ...] = sample[1:, ...].round()
        return sample

  
class CustomRandomResizedCrop(object):
    def __init__(self, p, scale=(0.3, 1.0)):
        self.p = p
        self.scale = scale

    def __call__(self, sample):
        #input shape: (C, H, W)
        _, H, W = sample.shape
        randn = np.random.rand()
        random_resizedcrop = transforms.RandomResizedCrop(
            (H, W), scale=self.scale, interpolation=transforms.InterpolationMode.NEAREST)
        
        if randn < self.p: 
            sample = random_resizedcrop(sample)
            sample[1:, ...] = sample[1:, ...].round()
        return sample
        

def image_resize(image, size=(224, 224)):
    #image shape: C, H, W or N, C, H, W
    H, W = size
    assert len(image.shape) == 3 or len(image.shape) == 4, \
     'input image must be of shape  C, H, W or N, C, H, W'

    if isinstance(image, np.ndarray): image = torch.from_numpy(image) 
    res_model = transforms.Resize((H, W), interpolation=transforms.InterpolationMode.NEAREST)
    image = res_model(image)
    return image


def data_augmentation(**kwargs):

    defaultKwargs = {
        'shuffle_tranforms':True,
        'crop_p':0.5, 
        'rotation_p':0.5, 
        'Hflip_p':0.5, 
        'Vflip_p':0.5, 
        'blur_p':0.5, 
        'rotation_angle_range':(0, 60),
        'crop_scale':(0.3, 1.0),
        'blur_kernel_size':(5, 9),
        'blur_sigma':(0.1, 11),
        }

    kwargs = {**defaultKwargs, **kwargs}

    transform_list = [
      CustomRandomResizedCrop(p=kwargs['crop_p'], scale=kwargs['crop_scale']),
      CustomRandomRotation(p=kwargs['rotation_p'], angle_range=kwargs['rotation_angle_range']),
      transforms.RandomHorizontalFlip(kwargs['Hflip_p']),
      transforms.RandomVerticalFlip(kwargs['Vflip_p']),
      FirstChannelRandomGaussianBlur(kwargs['blur_p'], kernel_size=kwargs['blur_kernel_size'], sigma=kwargs['blur_sigma']),
    ]
    if kwargs['shuffle_tranforms']: random.shuffle(transform_list)
    T = transforms.Compose(transform_list)
    return T