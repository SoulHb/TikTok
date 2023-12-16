import os
import torch
import cv2
from torch.utils.data import Dataset


class TikTok(Dataset):
    def __init__(self, images_dir, masks_dir, transform):
        """
               Init method for Custom PyTorch Dataset for TikTok Examples and masks.

                Args:
                    images_dir (str): Directory containing the Examples.
                    masks_dir (str): Directory containing the masks.
                    transform (dict): Dictionary with image and mask transformations.
                Return: None
                """
        super(TikTok, self).__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.list_images = sorted(os.listdir(self.images_dir))
        self.list_masks = sorted(os.listdir(self.masks_dir))
        self.transform = transform

    def __len__(self):
        """
                Get the number of Examples in the dataset.
                Args: None
                Returns:
                    int: Number of Examples in the dataset.
                """
        return len(self.list_images)

    def __getitem__(self, idx):
        """
                Get an item (image and mask) from the dataset at the specified index.

                Args:
                    idx (int): Index of the item.

                Returns:
                    tuple: (image, mask) pair.
                """
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

