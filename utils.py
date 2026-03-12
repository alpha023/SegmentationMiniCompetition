import torch
import torch.nn as nn

NUM_CLASSES = 21

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

