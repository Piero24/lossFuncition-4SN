### LOSS FUNTION PER HARDNET
# https://github.com/james128333/HarDNet-MSEG

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from nnunet.utilities.nd_softmax import softmax_helper
from scipy.ndimage import distance_transform_edt as dist
# from skimage.metrics import structural_similarity as SSIM

"""
This module contains various loss functions commonly used in deep learning models for segmentation tasks.
The code for these loss functions is adapted from the loss function library by bigironsphere on Kaggle.
    
    # From https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch

To use the bce_loss function in a training script (train.py), first import it from this module as follows:

Example:
    from loss import bce_loss

Then, in the line where you calculate the loss for your model (loss5 = structure_loss(lateral_map_5, gts)), 
    use the bce_loss function with the appropriate input arguments.

Example:
    loss5 = bce_loss(lateral_map_5, gts)

where lateral_map_5 is the predicted output of your model and gts is the ground truth segmentation mask.
"""



def bce_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Computes the binary cross entropy loss between the predictions 
        and the ground truth masks.

    Args:
        pred (torch.Tensor): the predicted logits (before sigmoid) of shape (N, C, H, W).
        mask (torch.Tensor): the ground truth binary masks of shape (N, C, H, W).

    Returns:
        torch.Tensor: the binary cross entropy loss tensor of shape (N, C, H, W).

    """

    bce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')

    return bce



def dice_loss(pred: torch.Tensor, mask: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Calculates the Dice loss between the predicted and target masks.

    Args:
        pred: Tensor of predicted masks with shape (N, C, H, W).
        mask: Tensor of target masks with shape (N, C, H, W).
        smooth: A smoothing factor to avoid division by zero.

    Returns:
        Tensor representing the Dice loss.

    """

    # pred = torch.sigmoid(pred)
    # inter = (pred * mask).sum(dim=(2, 3))
    # total = (pred + mask).sum(dim=(2, 3))
    # dice = 1 - (2*inter + smooth) / (total + smooth)

    # Comment out if your model contains a sigmoid or equivalent activation layer
    pred = F.sigmoid(pred)

    # Flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)

    inter = (pred * mask).sum()
    dice = 1 - ((2.*inter + smooth)/(pred.sum() + mask.sum() + smooth))

    return dice



def IoU_loss(pred: torch.Tensor, mask: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Calculate IoU loss between model predictions and ground truth masks.
    
     Args:
         pred (torch.Tensor): The predictions of the size model [batch_size, channels, height, width].
         mask (torch.Tensor): The ground truth masks of size [batch_size, channels, height, width].
         smooth (float, optional): The smooth value to use in the IoU formula. Defaults: 1e-6.
        
     Returns:
         torch.Tensor: The IoU loss value between predictions and ground truth masks.

    """

    pred = F.sigmoid(pred)

    # Flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)

    inter = (pred * mask).sum()
    total = (pred + mask).sum()
    union = total - inter
    IoU = 1 - ((inter + smooth) / (union + smooth))

    return IoU



def dice_bce_loss(pred: torch.Tensor, mask: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Computes the sum of Binary Cross Entropy (BCE) loss and Dice 
        loss between the predicted and target masks.
    
    Args:
        pred (torch.Tensor): The predicted mask tensor of shape (N, C, H, W)
        mask (torch.Tensor): The target mask tensor of shape (N, C, H, W)
        smooth (float, optional): A small float value to avoid zero division. Default is 1e-6.
    
    Returns:
        torch.Tensor: The combined loss tensor, which is the sum of BCE and Dice losses.
    
    """

    pred = F.sigmoid(pred)

    # Flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)

    inter = (pred * mask).sum()
    dice = 1 - ((2.*inter + smooth)/(pred.sum() + mask.sum() + smooth))
    bce = F.binary_cross_entropy(pred, mask, reduction='mean')

    return bce+dice



def log_cosh_dice_loss(pred: torch.Tensor, mask: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Computes the Log-Cosh Dice Loss between the predicted and target masks.

    Args:
        pred (torch.Tensor): The predicted mask.
        mask (torch.Tensor): The target mask.
        smooth (float): A small constant added to the denominator to avoid division by zero.

    Returns:
        torch.Tensor: The computed Log-Cosh Dice Loss.
    
    Notes:
        Survey Paper DOI: 10.1109/CIBCB48159.2020.9277638
        Software Release DOI: https://doi.org/10.1016/j.simpa.2021.100078
        Adapted from https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions

    """

    pred = F.sigmoid(pred)

    # Flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)

    inter = (pred * mask).sum()
    dice = 1 - ((2.*inter + smooth)/(pred.sum() + mask.sum() + smooth))

    return torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)



def focal_loss(pred: torch.Tensor, mask: torch.Tensor, alpha: float = 0.8, 
               gamma: float = 2, smooth: float = 1e-6) -> torch.Tensor:
    """Computes focal loss between predicted and ground truth masks.
    
    Args:
        pred (torch.Tensor): Predicted mask of shape (batch_size, channels, height, width).
        mask (torch.Tensor): Ground truth mask of shape (batch_size, channels, height, width).
        alpha (float): Weighting factor to balance positive and negative class samples. Default: 0.8.
        gamma (float): Modulating factor to down-weight easy examples (samples with high confidence). Default: 2.
        smooth (float): Smoothing factor to avoid division by zero. Default: 1e-6.
    
    Returns:
        torch.Tensor: Focal loss scalar value.

    """

    pred = F.sigmoid(pred)

    # Flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)

    bce = F.binary_cross_entropy(pred, mask, reduction='mean')
    bce_exp = torch.exp(-bce)
    focal_loss = alpha * (1-bce_exp)**gamma * bce

    return focal_loss



def tversky_loss(pred: torch.Tensor, mask: torch.Tensor, alpha: float = 0.3, 
                 beta: float = 0.7, smooth: float = 1e-6) -> torch.Tensor:
    
    """Calculate the Tversky loss between the reference segmentation mask and the 
        predicted segmentation mask.

    Args:
        pred (torch.Tensor): Tensor of shape (B, C, H, W) representing the predicted segmentation mask.
        mask (torch.Tensor): Tensor of shape (B, C, H, W) which represents the reference segmentation mask.
        alpha (float, optional): The weight assigned to false positives. Defaults: 0.3.
        beta (float, optional): The weight assigned to false negatives. Defaults: 0.7.
        smooth (float, optional): The smooth constant to avoid division by zero. Defaults: 1e-6.

    Returns:
        torch.Tensor: A tensor with a single value representing the Tversky loss.

    """

    pred = F.sigmoid(pred)

    # Flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)

    # True Positives, False Positives & False Negatives
    TP = (pred * mask).sum()
    FP = (pred * (1-mask)).sum()
    FN = ((1-pred) * mask).sum()

    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)

    return 1 - Tversky



def focal_tversky_loss(pred: torch.Tensor, mask: torch.Tensor, alpha: float = 0.3, 
                       beta: float = 0.7, gamma: float = 1, smooth: float = 1e-6) -> torch.Tensor:
    
    """Compute the Focal Tversky loss between the predicted segmentation 
        mask and the ground truth mask.

    Args:
        pred (torch.Tensor): the predicted segmentation mask
        mask (torch.Tensor): the ground truth mask
        alpha (float): weight of false positives
        beta (float): weight of false negatives
        gamma (float): weight of the Focal Tversky loss
        smooth (float): smoothing factor to avoid division by zero

    Returns:
        torch.Tensor: the Focal Tversky loss between the predicted segmentation mask 
            and the ground truth mask
    """

    pred = F.sigmoid(pred)

    # Flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)

    # True Positives, False Positives & False Negatives
    TP = (pred * mask).sum()
    FP = (pred * (1-mask)).sum()
    FN = ((1-pred) * mask).sum()

    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
    FocalTversky = (1 - Tversky)**gamma

    return FocalTversky



def combo_loss(pred: torch.Tensor, mask: torch.Tensor, alpha: float = 0.5, 
               ce_ratio: float = 0.5, eps: float = 1e-9, smooth: float = 1e-6) -> torch.Tensor:
    """Calculate the Combo loss function for training image segmentation models.

    Subjects:
        pred (torch.Tensor): Tensor of model predictions.
        mask (torch.Tensor): Tensor of correct binary masks of image labels.
        alpha (float, optional): Weighing value between Cross-Entropy loss and F1-score loss. 
            The default is 0.5.

        ce_ratio (float, optional): Weighting value between Combo loss and Dice loss.
            The default is 0.5.

        eps (float, optional): Minimum value for clamping prediction values.
            The default is 1e-9.

        smooth (float, optional): Smoothing factor to avoid numerical instability problems.
            The default is 1e-6.

    Come back:
        torch.Tensor: The weighted sum of the Cross-Entropy loss and the F1-score loss.

    Notes:
        ALFA = 0.5 # < 0.5 penalizes FP more, > 0.5 penalizes FN more
        CE_RATIO = 0.5 #weighted contribution of modified CE loss to dice loss

    WARNING: IT DOES NOT WORK

    """

    pred = F.sigmoid(pred)

    # Flatten label and prediction tensors
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



def structure_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Calculate the loss function Structure for training image segmentation models.
    
     Args:
         pred (torch.Tensor): Tensor of model predictions.
         mask (torch.Tensor): Tensor of correct binary masks of image labels.
    
     Returns:
         torch.Tensor: The average of the sum of the weighted binary cross-entropy 
            loss (wbce) and the Weighted Intersection over Union loss (wiou).
     """
    
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()



class RWLoss(nn.Module):


    # RRW Maps
    def __init__(self):
        super(RWLoss, self).__init__()
    
    # Represents the forwarding method of the RWLoss object instance. 
    # This function takes two tensors x and y_ as input and returns a 
    # loss value calculated on the basis of these tensors.
    def forward(self, x, y_):
        # Python library for reading and writing image data in various 
        # medical formats, such as the NIfTI format.
        import nibabel as nib

        # Softmax
        #
        # The input array x is normalized using the Softmax function, 
        # which returns a new array of normalized probability values.
        x = softmax_helper(x)

        # One hot conversion
        #
        # A new tensor y of equal size to x is created, 
        # with all values initialized to zero.
        y = torch.zeros(x.shape)

        # If the GPU is available
        if x.device.type == "cuda":
            # The tensor y is moved to the GPU 
            # to take advantage of GPU acceleration during computation.
            y = y.cuda(x.device.index)
        
        # A value of 1 is set for each index indicated in y_, 
        # where y_ is a class label index tensor.
        y.scatter_(1, y_.long(), 1)

        # The tensor y is moved to the CPU and converted into a NumPy array, 
        # which is needed for calculating the RRW maps.
        y_cpu = y.detach().cpu().numpy()
        
        """
        path = "/home/miguelv/nnunet/debug/"
        for i in range(y_cpu.shape[0]):
           data = np.moveaxis(y_cpu[i], 0, -1)
           nib.save(nib.Nifti1Image(data, np.eye(4)), path + "Y-" + str(i+1) + ".nii.gz")
        """

        # Probably move `y` to CPU
        #
        # A new NumPy rrwmap array equal in size to y_cpu is created, 
        # with all values initialized to zero.
        rrwmap = np.zeros_like(y_cpu)

        # A for loop is performed on the batches of data.
        for b in range(rrwmap.shape[0]):
            # A for loop is performed on the tensor channels, i.e. on the label classes.
            for c in range(rrwmap.shape[1]):
                # The distance between the normalized probability vector y_cpu[b, c] 
                # and the uniform distribution is calculated.
                rrwmap[b, c] = dist(y_cpu[b, c])
                # The distance is normalized and inverted, 
                # assigning a value of 1 to the classes most similar to the 
                # uniform distribution and a value of 0 to the classes least 
                # similar to the uniform distribution.
                rrwmap[b, c] = -1 * (rrwmap[b, c] / (np.max(rrwmap[b, c] + 1e-15)))

        # Values of 0 in the rrwmap array are replaced 
        # with 1 to avoid the NaN loss value.
        rrwmap[rrwmap==0] = 1

        """
        for i in range(rrwmap.shape[0]):
           data = np.moveaxis(rrwmap[i], 0, -1)
           nib.save(nib.Nifti1Image(data, np.eye(4)), path + "M-" + str(i+1) + ".nii.gz")
        """

        # The NumPy rrwmap array is converted to a PyTorch tensor.
        rrwmap = torch.Tensor(rrwmap)

        # If the GPU is available
        if x.device.type == "cuda":
            # The rrwmap tensor is moved to the GPU to take advantage 
            # of GPU acceleration during computation.
            rrwmap = rrwmap.cuda(x.device.index)
        
        # Calculates the loss value (loss) as the mean of the 
        # product between the tensors x and rrwmap.
        #
        # In other words, the idea is to penalize wrong predictions 
        # more that are farther from the uniform distribution than 
        # correct predictions that are more similar to the uniform distribution. 
        # In this way, the loss function favors the convergence towards a 
        # uniform distribution and therefore a better generalization of the model.
        loss = torch.mean(x * rrwmap)

        """
        print(loss)
        if torch.isnan(loss):
           raise Exception("para")
        """

        return loss
 