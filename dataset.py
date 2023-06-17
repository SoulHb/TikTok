import torchvision
import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2


class TikTok(Dataset):
    def __init__(self, images_dir, masks_dir, transform):
        super(TikTok, self).__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.list_images = sorted(os.listdir(self.images_dir))
        self.list_masks = sorted(os.listdir(self.masks_dir))
        self.transform = transform

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.list_images[idx])
        mask_path = os.path.join(self.masks_dir, self.list_masks[idx])
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image/255
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        mask[mask == 255.0] = 1.0
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if self.transform:
            transformed_image = self.transform['image'](image=image)
            transformed_mask = self.transform['mask'](image=mask)
            image = transformed_image['image']
            mask = transformed_mask['image']
        return image, mask.to(torch.float32)

