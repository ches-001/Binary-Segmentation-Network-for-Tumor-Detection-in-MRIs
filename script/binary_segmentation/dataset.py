import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, images, images_df, transform=None, tp=0.5):
        self.images = images
        self.images_df = images_df
        self.transform = transform
        self.tp = tp
        
        
    def __len__(self):
        return len(self.images_df)
    
       
    def __getitem__(self, idx:int):
        image = self.images[idx]
        image = np.expand_dims(image, axis=0)
        
        lb_seg_mask = self.rle2mask(image.shape[1:], self.images_df.large_bowel[idx])
        sb_seg_mask = self.rle2mask(image.shape[1:], self.images_df.small_bowel[idx])
        stomach_seg_mask = self.rle2mask(image.shape[1:], self.images_df.stomach[idx])
        
        gt_mask = np.stack((lb_seg_mask, sb_seg_mask, stomach_seg_mask), axis=0)

        image = self.resize(image.astype('float32')) #shape: (1, H, W)
        gt_mask = self.resize(gt_mask.astype('float32')) #shape: (3, H, W)

        if self.transform:
            randn = np.random.rand()
            if randn < self.tp:
                aug = self.transform(
                    torch.from_numpy(
                        np.concatenate((image, gt_mask), axis=0)
                    ))
                image, gt_mask = aug[0].unsqueeze(dim=0), aug[1:]

        image = self.image_normalize(image)
        return image, gt_mask
    
    
    def image_normalize(self, image):
        #image size: (C, H, W)
        image = image.max() - image
        image = image - image.min()
        image = image / image.max()
        return image


    def rle2mask(self, img_shape, rle:str):
        #correct order: (H, W)
        H, W = img_shape
        if pd.isnull(rle): return np.zeros((H, W), dtype=np.int8)
        
        rle = rle.split()
        start_idx, lengths = np.array(rle[0::2], dtype=int), np.array(rle[1::2], dtype=int)
        end_idx = start_idx + (lengths - 1)

        mask = np.zeros(H*W, dtype=np.int8)
        for start, end in zip(start_idx, end_idx):
            mask[start:end] = 1

        mask = mask.reshape(H, W)
        return mask
    
    
    def resize(self, image, size=(224, 224)):
        #image shape: C, H, W or N, C, H, W
        H, W = size
        assert len(image.shape) == 3 or len(image.shape) == 4, 'input image must be of shape  C, H, W or N, C, H, W'

        if isinstance(image, np.ndarray): image = torch.from_numpy(image) 
        res_model = transforms.Resize((H, W), interpolation=transforms.InterpolationMode.NEAREST)
        image = res_model(image)
        return image