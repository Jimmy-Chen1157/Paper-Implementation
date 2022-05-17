import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class WaterBodies(Dataset):

    def __init__(self, csv_file, transform=None):
        super().__init__()
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_path = self.annotations.iloc[index, 1]
        mask_path = self.annotations.iloc[index, 2]
        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("1"), dtype=np.float32)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask.unsqueeze(0)
        return image, mask


if __name__ == "__main__":
    train_csv = "datasets/Water Bodies/train/train_csv.csv"
    val_csv = "datasets/Water Bodies/val/val_csv.csv"
    transform = A.Compose([
        A.Resize(height=256, width=256),
        ToTensorV2(),
    ])
    data = WaterBodies(train_csv, transform)
    # for img, mask in data:
    #     print(img.shape, mask.shape)
    # print(img.dtype, mask.dtype)
    print(len(data))
