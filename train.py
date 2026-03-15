import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

from dataset import VOCDataset
from model import LightweightUNet
from utils import dice_score,DiceLoss, log_epoch, log_experiment_start,log_final_results
from early_stopping import EarlyStopping

from secret import compute_flops, ranking_score
from parameters import *

import random
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

image_transform = T.Compose([
    # T.Resize((320,320)),
    # T.RandomCrop((300,300)),
    T.Resize((300,300)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2,0.2,0.2,0.05),
    T.ToTensor()
])

mask_transform = T.Compose([
    # T.Resize((320,320), interpolation=T.InterpolationMode.NEAREST),
    # T.RandomCrop((300,300)),
    T.Resize((300,300)),
    T.RandomHorizontalFlip()
])

dataset = VOCDataset(
    IMAGE_DIR,
    MASK_DIR,
    image_transform=image_transform,
    mask_transform=mask_transform
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# train_dataset, val_dataset = random_split(dataset,[train_size,val_size])

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE)

model = LightweightUNet(num_classes=21).to(DEVICE)
flops_giga, params = compute_flops(model, device=DEVICE)
log_experiment_start("LightweightUNet", flops_giga, params)

early_stopping = EarlyStopping(PATIENCE)

print(f"Model FLOPs: {flops_giga:.3f} GFLOPs")
print(f"Parameters: {params/1e6:.2f} Million")

criterion = torch.nn.CrossEntropyLoss(ignore_index=255,label_smoothing=0.1)
dice_loss=DiceLoss()

optimizer = torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

def loss_fn(pred, target):
    return 0.7*criterion(pred, target) + 0.3*dice_loss(pred, target)

for epoch in range(EPOCHS):

    # ---------------- TRAIN ----------------
    model.train()
    total_loss = 0
    train_dice_total = 0

    for images, masks in train_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)

        loss = loss_fn(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # compute train dice
        dice = dice_score(outputs.detach(), masks)
        train_dice_total += dice

    avg_train_dice = train_dice_total / len(train_loader)

    scheduler.step()

    avg_loss = total_loss / len(train_loader)

    # ---------------- VALIDATION ----------------
    model.eval()
    dice_total = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            dice = dice_score(outputs, masks)

            dice_total += dice

    avg_dice = dice_total / len(val_loader)

    current_lr = optimizer.param_groups[0]['lr']

    # print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_loss:.4f} | Val Dice: {avg_dice:.4f} | LR: {current_lr:.6f}")
    log_epoch(epoch+1, EPOCHS, avg_loss, avg_dice, current_lr,avg_train_dice)
    early_stopping(avg_dice, model)

    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break

score = ranking_score(avg_dice, flops_giga)
log_final_results(avg_dice, score)

with open("score.txt", "w") as f:
    f.write(f"{avg_dice:.6f}")

torch.save(model.state_dict(),"model.pth")
