import torch
import torch.nn as nn

NUM_CLASSES = 21
from parameters import *

from datetime import datetime

LOG_FILE = "training_log.txt"

def log_to_file(text, file=LOG_FILE):
    """
    Appends a line to the log file
    """
    print(text)  # also print to terminal
    with open(file, "a") as f:
        f.write(text + "\n")

def log_experiment_start(model_name, flops, params):

    log_to_file("\n" + "="*60)
    log_to_file(f"MODEL: {model_name}")
    log_to_file(f"Run Time: {datetime.now()}")
    log_to_file("="*60)

    log_to_file("PARAMETERS")

    log_to_file(f"IMAGE_DIR: {IMAGE_DIR}")
    log_to_file(f"MASK_DIR: {MASK_DIR}")
    log_to_file(f"BATCH_SIZE: {BATCH_SIZE}")
    log_to_file(f"EPOCHS: {EPOCHS}")
    log_to_file(f"LEARNING_RATE: {LEARNING_RATE}")
    log_to_file(f"PATIENCE: {PATIENCE}")
    log_to_file(f"DEVICE: {DEVICE}")
    log_to_file(f"FLOPS: {flops:.3f} GFLOPs")
    log_to_file(f"PARAMETERS: {params/1e6:.2f} Million")

    log_to_file("")
    log_to_file("TRAINING LOG")
    log_to_file("-"*60)


def log_epoch(epoch, epochs, train_loss, val_dice, lr,avg_train_dice):

    log_line = f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Train Dice: {avg_train_dice:.4f} | Val Dice: {val_dice:.4f} | LR: {lr:.6f}"

    print(log_line)        # print to terminal
    log_to_file(log_line)  # save to file

def log_final_results(avg_dice, score):

    log_to_file("-"*60)
    log_to_file(f"FINAL VAL DICE: {avg_dice:.6f}")
    log_to_file(f"RANKING SCORE: {score:.6f}")
    log_to_file("="*60)
    
def dice_score(pred,target):

    pred = torch.argmax(pred,dim=1)

    dice = 0
    count = 0

    for cls in range(NUM_CLASSES):

        pred_i = (pred==cls).float()
        target_i = (target==cls).float()

        mask = target != 255

        pred_i *= mask
        target_i *= mask

        intersection = (pred_i*target_i).sum()
        union = pred_i.sum()+target_i.sum()

        if union == 0:
            continue

        dice += (2*intersection)/(union+1e-6)
        count += 1

    return dice/count


class DiceLoss(nn.Module):

    def __init__(self,smooth=1):
        super().__init__()
        self.smooth=smooth

    def forward(self, pred, target):

        pred = torch.softmax(pred, dim=1)
        pred=torch.argmax(pred,dim=1)

        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred == target).float().sum()

        dice = (2 * intersection + self.smooth) / (pred.numel() + target.numel() + self.smooth)

        return 1 - dice
