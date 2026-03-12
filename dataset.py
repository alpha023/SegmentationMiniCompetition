import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from parameters import IMAGE_SIZE


class VOCDataset(Dataset):

    def __init__(self, image_dir, mask_dir,image_transform=None, mask_transform=None):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.masks = sorted(os.listdir(mask_dir))
        self.images = [m.replace(".png", ".jpg") for m in self.masks]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # if self.transform:
        #     image = self.image_transform(image)

        # mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)

        if self.image_transform and self.mask_transform:

            seed = torch.randint(0,100000,(1,)).item()

            torch.manual_seed(seed)
            image = self.image_transform(image)

            torch.manual_seed(seed)
            mask = self.mask_transform(mask)

        else:
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
            mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)

        mask = torch.from_numpy(np.array(mask)).long()

        mask[mask > 20] = 255

        return image, mask