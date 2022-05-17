import os
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import WaterBodies
from unet import UNet


device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PTH = "Paper Implementation/U-Net/WBsegmentation.pth"
VAL_CSV = "datasets/Water Bodies/val/val_csv.csv"
SAVED_IMG_PTH = "Paper Implementation/U-Net/saved_img"
BATCH_SIZE = 16


model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PTH))

transform = A.Compose([
    A.Resize(height=256, width=256),
    ToTensorV2(),
])


val_set = WaterBodies(VAL_CSV, transform)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)


def save_pred_img(model, loader):
  model.eval()
  with torch.no_grad():
    for idx, (x, y) in enumerate(loader):
      x = x.to(device)
      y = y.to(device)
      pred = torch.sigmoid(model(x))
      pred = (pred > 0.5).float()
      pred_path = os.path.join(SAVED_IMG_PTH, "pred", 'pred_' + str(idx) + '.jpg')
      mask_path = os.path.join(SAVED_IMG_PTH, "mask", 'mask_' + str(idx) + '.jpg')
      data_path = os.path.join(SAVED_IMG_PTH, "data", 'data_' + str(idx) + '.jpg')
      save_image(pred, pred_path)
      save_image(y, mask_path)
      save_image(x, data_path)
  model.train()


def check_accuracy(model, loader):
  num_correct = 0
  num_pixels = 0
  model.eval()
  with torch.no_grad():
    for x, y in loader:
      x = x.to(device)
      y = y.to(device)
      preds = torch.sigmoid(model(x))
      preds = (preds > 0.5).float()
      num_correct += (preds == y).sum()
      num_pixels += torch.numel(preds)
  print(
      f"Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}"
  )
  model.train()


check_accuracy(model, val_loader)
save_pred_img(model, val_loader)
