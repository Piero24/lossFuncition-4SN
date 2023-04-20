#LOSS FUNTION PER HARDNET
#https://github.com/james128333/HarDNet-MSEG

import torch
import torch.nn.functional as F
#from skimage.metrics import structural_similarity as SSIM

# From https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch


##nel file train.py importare le loss ...
#from loss import bce_loss
##e usare la nuova funzione di loss nella riga : 
#loss5 = structure_loss(lateral_map_5, gts)
##es:
#loss5 = bce_loss(lateral_map_5, gts)



def bce_loss(pred, mask):

    bce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')

    return bce


def dice_loss(pred, mask, smooth=1e-6):

    #pred = torch.sigmoid(pred)
    #inter = (pred * mask).sum(dim=(2, 3))
    #total = (pred + mask).sum(dim=(2, 3))
    #dice = 1 - (2*inter + smooth) / (total + smooth)
    # comment out if your model contains a sigmoid or equivalent activation layer
    pred = F.sigmoid(pred)

    # flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)

    inter = (pred * mask).sum()
    dice = 1 - ((2.*inter + smooth)/(pred.sum() + mask.sum() + smooth))

    return dice



def IoU_loss(pred, mask, smooth=1e-6):

    pred = F.sigmoid(pred)

    # flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)
    inter = (pred * mask).sum()
    total = (pred + mask).sum()
    union = total - inter
    IoU = 1 - ((inter + smooth) / (union + smooth))
    return IoU


def dice_bce_loss(pred, mask, smooth=1e-6):
    pred = F.sigmoid(pred)
    # flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)
    inter = (pred * mask).sum()
    dice = 1 - ((2.*inter + smooth)/(pred.sum() + mask.sum() + smooth))
    bce = F.binary_cross_entropy(pred, mask, reduction='mean')
    return bce+dice

#Survey Paper DOI: 10.1109/CIBCB48159.2020.9277638
#Software Release DOI: https://doi.org/10.1016/j.simpa.2021.100078
#Adapted from https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions
def log_cosh_dice_loss(pred, mask, smooth=1e-6):
    pred = F.sigmoid(pred)
    # flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)
    inter = (pred * mask).sum()
    dice = 1 - ((2.*inter + smooth)/(pred.sum() + mask.sum() + smooth))
    return torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)

def focal_loss(pred, mask, alpha=0.8, gamma=2, smooth=1e-6):
    pred = F.sigmoid(pred)
    # flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)
    bce = F.binary_cross_entropy(pred, mask, reduction='mean')
    bce_exp = torch.exp(-bce)
    focal_loss = alpha * (1-bce_exp)**gamma * bce

    return focal_loss


def tversky_loss(pred, mask, alpha=0.3, beta=0.7, smooth=1e-6):
    pred = F.sigmoid(pred)
    # flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)
    # True Positives, False Positives & False Negatives
    TP = (pred * mask).sum()
    FP = (pred * (1-mask)).sum()
    FN = ((1-pred) * mask).sum()
    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
    return 1 - Tversky


def focal_tversky_loss(pred, mask, alpha=0.3, beta=0.7, gamma=1, smooth=1e-6):
    pred = F.sigmoid(pred)
    # flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)
    # True Positives, False Positives & False Negatives
    TP = (pred * mask).sum()
    FP = (pred * (1-mask)).sum()
    FN = ((1-pred) * mask).sum()
    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
    FocalTversky = (1 - Tversky)**gamma
    return FocalTversky

#non funziona
def combo_loss(pred, mask,  alpha=0.5, ce_ratio=0.5, eps=1e-9, smooth=1e-6):
    # ALFA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
    # CE_RATIO = 0.5 #weighted contribution of modified CE loss compared to Dice loss
    pred = F.sigmoid(pred)
    # flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)
    # True Positives, False Positives & False Negatives
    inter = (pred * mask).sum()
    dice = (2.*inter + smooth)/(pred.sum() + mask.sum() + smooth)
    bce = F.binary_cross_entropy(pred, mask, reduction='mean')

    pred = torch.clamp(pred, eps, 1.0 - eps)
    out = - (alpha * ((mask * torch.log(pred)) +
                      ((1 - alpha) * (1.0 - mask) * torch.log(1.0 - pred))))
    weighted_ce = out.mean(-1)
    combo = (ce_ratio * weighted_ce) - ((1 - ce_ratio) * dice)

    return combo


def structure_loss(pred, mask):

    weit = 1 + 5 * \
        torch.abs(F.avg_pool2d(mask, kernel_size=31,
                               stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)

    return (wbce + wiou).mean()
