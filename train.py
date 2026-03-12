import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

from dataset import VOCDataset
from model import LightSegNet
from utils import dice_score,DiceLoss
from early_stopping import EarlyStopping

from secret import compute_flops, ranking_score
from parameters import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

image_transform = T.Compose([
    T.Resize((320,320)),
    T.RandomCrop((300,300)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2,0.2,0.2,0.05),
    T.ToTensor()
])

mask_transform = T.Compose([
    T.Resize((320,320), interpolation=T.InterpolationMode.NEAREST),
    T.RandomCrop((300,300)),
    T.RandomHorizontalFlip()
])

# image_dir = "dataset/VOC2012_train_val/JPEGImages"
# mask_dir = "dataset/VOC2012_train_val/SegmentationClass"

# dataset = VOCDataset(image_dir, mask_dir, transform)
dataset = VOCDataset(
    IMAGE_DIR,
    MASK_DIR,
    image_transform=image_transform,
    mask_transform=mask_transform
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset,[train_size,val_size])

train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE)

model = LightSegNet().to(DEVICE)
flops_giga, params = compute_flops(model, device=DEVICE)
early_stopping = EarlyStopping(PATIENCE)

print(f"Model FLOPs: {flops_giga:.3f} GFLOPs")
print(f"Parameters: {params/1e6:.2f} Million")

criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
dice_loss=DiceLoss()

optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

def loss_fn(pred, target):
    return 0.3*criterion(pred, target) + dice_loss(pred, target)

for epoch in range(EPOCHS):

    # ---------------- TRAIN ----------------
    model.train()
    total_loss = 0

    for images, masks in train_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)

        loss = loss_fn(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

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

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {avg_loss:.4f}")
    print(f"Validation Dice: {avg_dice:.4f}")
    print(f"Learning Rate: {current_lr:.6f}")
    print("----------------------------------")
    early_stopping(avg_dice, model)

    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break

score = ranking_score(avg_dice, flops_giga)
print(f"Ranking Score (Dice/FLOPs): {score:.6f}")

with open("score.txt", "w") as f:
    f.write(f"{avg_dice:.6f}")

torch.save(model.state_dict(),"model.pth")

# import torch
# import torchvision.transforms as T
# from torch.utils.data import DataLoader, random_split

# from dataset import VOCDataset
# from model import MobileNetUNet
# from utils import dice_score

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# transform = T.Compose([
#     T.Resize((300,300)),
#     T.ToTensor()
# ])

# image_dir = "dataset/VOC2012_train_val/JPEGImages"
# mask_dir = "dataset/VOC2012_train_val/SegmentationClass"

# dataset = VOCDataset(image_dir, mask_dir, transform)

# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size

# train_dataset, val_dataset = random_split(dataset,[train_size,val_size])

# train_loader = DataLoader(train_dataset,batch_size=8,shuffle=True)
# val_loader = DataLoader(val_dataset,batch_size=8)

# model = MobileNetUNet().to(DEVICE)

# criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

# optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

# EPOCHS = 30

# for epoch in range(EPOCHS):

#     model.train()

#     for images,masks in train_loader:

#         images = images.to(DEVICE)
#         masks = masks.to(DEVICE)

#         outputs = model(images)
#         print(f"Mask:{outputs.shape}")
#         print(f"Images:{outputs.shape}")
#         print("Uniqueness",torch.unique(outputs[0]))

#         loss = criterion(outputs,masks)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         print(f"Mask:{masks.shape}")
#         print(f"Images:{images.shape}")
#         print("Uniqueness",torch.unique(masks[0]))
#         print("=====")
#         break
    

#     print("Epoch",epoch)

# torch.save(model.state_dict(),"model.pth")