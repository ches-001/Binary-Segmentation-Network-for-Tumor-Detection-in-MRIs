import numpy as np
import pandas as pd
import torch, random
from torch.utils.data import Dataset
from .transforms import image_resize, image_normalize
from typing import Optional


class ImageDataset(Dataset):
    def __init__(self, images:list, images_df:pd.DataFrame, transform:Optional[object]=None, 
                 tp:float=0.5, input_size:tuple=(224, 224), target_size:tuple=(224, 224)):
      
        self.images = images
        self.images_df = images_df
        self.transform = transform
        self.tp = tp
        self.input_size = input_size
        self.target_size = target_size
        
        
    def __len__(self):
        return len(self.images_df)
    
       
    def __getitem__(self, idx:int):
        image = self.images[idx]
        image = np.expand_dims(image, axis=0).astype('int32')
        
        lb_seg_mask = self.rle2mask(image.shape[1:], self.images_df.large_bowel[idx])
        sb_seg_mask = self.rle2mask(image.shape[1:], self.images_df.small_bowel[idx])
        stomach_seg_mask = self.rle2mask(image.shape[1:], self.images_df.stomach[idx])
        
        gt_mask = np.stack((lb_seg_mask, sb_seg_mask, stomach_seg_mask), axis=0).astype('int32')

        if self.transform:
            randn = random.random()
            if randn < self.tp:
                aug = self.transform(
                    torch.from_numpy(
                        np.concatenate((image, gt_mask), axis=0)
                    ))
                image, gt_mask = aug[0].unsqueeze(dim=0), aug[1:]

        image = image_resize(image, self.input_size).type(torch.float32) #shape: (1, H, W)
        gt_mask = image_resize(gt_mask, self.target_size).type(torch.float32) #shape: (3, H, W)
        image = image_normalize(image)

        return image, gt_mask
    

    def rle2mask(self, img_shape:tuple, rle:str):
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