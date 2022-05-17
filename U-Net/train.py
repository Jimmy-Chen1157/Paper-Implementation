import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import WaterBodies
from unet import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter


TRAIN_CSV = "datasets/Water Bodies/train/train_csv.csv"
VAL_CSV = "datasets/Water Bodies/val/val_csv.csv"
SAVED_PTH = "Paper Implementation/U-Net/WBsegmentation.pth"
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 3e-4

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = A.Compose([
    A.Resize(height=256, width=256),
    ToTensorV2(),
])


train_set = WaterBodies(TRAIN_CSV, transform)
val_set = WaterBodies(VAL_CSV, transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)


model = UNet(in_channels=3, out_channels=1).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler()
writer = SummaryWriter(f"runs/WaterBody/WBsegmentation")


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


if __name__ == "__main__":
    for epoch in range(EPOCHS):
        loop = tqdm(train_loader)
        step = 0
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device)
            target = targets.to(device)

            with torch.cuda.amp.autocast():
                pred = model(data)
                loss = loss_fn(pred, target)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            writer.add_scalar("Training Loss", loss, global_step=step)
            step += 1

            loop.set_postfix(loss=loss.item())
        torch.save(model.state_dict(), SAVED_PTH)
        check_accuracy(model, val_loader)
