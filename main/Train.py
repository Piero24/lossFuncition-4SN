import os
import logging
import argparse
from datetime import datetime

import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

from lib.pvt import HSNet
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter

# Loss already present in the file
from loss import bce_loss, dice_loss, IoU_loss
from loss import dice_bce_loss, log_cosh_dice_loss
from loss import focal_loss, tversky_loss
from loss import focal_tversky_loss, combo_loss
from loss import structure_loss
from lossTest import RWLoss




def test(model: torch.nn.Module, path: str, dataset: str) -> float:
    """Compute the Dice Similarity Coefficient (DSC) on a dataset using a given model.

    Args:
        model: the PyTorch model to use for testing.
        path (str): the path to the directory containing the dataset.
        dataset (str): the name of the dataset to test.

    Returns:
        float: the average DSC value computed on the dataset.
    
    Notes:
        We have updated the function to stop the warning
        "nn.functional.upsample is deprecated. Use nn.functional.interpolate instead."
        which occurs when we use PyTorch's upsample function.
        The solution was to replace F.upsample with F.interpolate.

        Warning example:

            \\path\HSNet\PyTorch\env\Lib\site-packages\torch\nn\functional.py:3737:
            UserWarning: nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.
            warnings.warn("nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.")

        If you want to go back to using upsample just change interpolate
        with upsample without making any further changes to parameters or anything else.

    """

    # Build paths to images and masks
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)

    # Set model to evaluation mode
    model.eval()

    # Count number of images in the dataset
    num1 = len(os.listdir(gt_root))

    # Create dataset loader
    test_loader = test_dataset(image_root, gt_root, 352)

    # Initialize DSC variable
    DSC = 0.0

    # Compute DSC for each image in the dataset
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        
        res, res1, res2, res3  = model(image)

        # eval Dice
        # Upsample result and compute sigmoid
        res = F.interpolate(res + res1 + res2 + res3 , size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)

         # Compute Dice Similarity Coefficient
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice
    
    # Compute average DSC and return it
    return DSC / num1



def train(train_loader: torch.utils.data.DataLoader, model: torch.nn.Module, 
          optimizer: torch.optim.Optimizer, epoch: int, test_path: str) -> float:
    
    """Trains the model on the given train_loader using the specified optimizer and loss function.
        Also performs testing on multiple datasets to monitor the model's progress.

    Args:
        train_loader (DataLoader): the data loader for the training set
        model (nn.Module): the model to be trained
        optimizer (optim.Optimizer): the optimizer to use during training
        epoch (int): the current epoch number
        test_path (str): the path to the folder containing the testing datasets

    Returns:
        The average dice similarity coefficient (DSC) across all testing datasets
    
    Notes:
        We have updated the function to stop the warning
        "nn.functional.upsample is deprecated. Use nn.functional.interpolate instead."
        which occurs when we use PyTorch's upsample function.
        The solution was to replace F.upsample with F.interpolate.

        Warning example:

            \\path\HSNet\PyTorch\env\Lib\site-packages\torch\nn\functional.py:3737:
            UserWarning: nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.
            warnings.warn("nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.")

        If you want to go back to using upsample just change interpolate
        with upsample without making any further changes to parameters or anything else.

    """

    # set model to train mode
    model.train()

    global best

    # initialize size rates for data rescaling
    size_rates = [0.75, 1, 1.25]

    # initialize loss record
    loss_P2_record = AvgMeter()

    # iterate over training data batches
    for i, pack in enumerate(train_loader, start=1):

        # iterate over size rates for data rescaling
        for rate in size_rates:
            # zero the gradients in the optimizer
            optimizer.zero_grad()

            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()

            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            
            # ---- forward ----
            P1, P2, P3, P4= model(images)

            # ---- loss function ----
            # loss_P1 = structure_loss(P1, gts)
            # loss_P2 = structure_loss(P2, gts)
            # loss_P3 = structure_loss(P3, gts)
            # loss_P4 = structure_loss(P4, gts)
            # loss = loss_P1 + loss_P2 + loss_P3 + loss_P4
            # Creazione dell'istanza della classe RWLoss
            rw_loss = RWLoss()
            loss = rw_loss.forward(P1, gts)
            
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            
            # ---- recording loss ----
            #if rate == 1:
                #loss_P2_record.update(loss_P4.data, opt.batchsize)
        
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}] lr'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_P2_record.show()), optimizer.param_groups[0]['lr'])
            
    # save model
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypPVT.pth')

    # choose the best model
    global dict_plot
    test1path = '//NAS_home/Develop/Coding/Research/HSNet/dataset/TestDataset/'
    if (epoch + 1) % 1 == 0:

        for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
            dataset_dice = test(model, test1path, dataset)
            logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
            print(dataset, ': ', dataset_dice)
            dict_plot[dataset].append(dataset_dice)

        meandice = test(model, test_path, 'test')
        dict_plot['test'].append(meandice)

        if meandice > best:
            best = meandice
            torch.save(model.state_dict(), save_path + 'PolypPVT.pth')
            torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypPVT-best.pth')
            print('##############################################################################best', best)
            logging.info('##############################################################################best:{}'.format(best))



def plot_train(dict_plot: dict = None, name: list = None) -> None:
    """Plot the training curves for different datasets and save the resulting image as "eval.png".

    Args:
        dict_plot (dict): A dictionary containing the datasets names as keys 
            and the corresponding training curves as values.

        name (list): A list of the names of the datasets to be plotted.

    """

    color = ['red', 'lawngreen', 'lime', 'gold', 'm', 'plum', 'blue']
    line = ['-', "--"]

    for i in range(len(name)):
        plt.plot(dict_plot[name[i]], label=name[i], color=color[i], linestyle=line[(i + 1) % 2])
        transfuse = {'CVC-300': 0.902, 'CVC-ClinicDB': 0.918, 'Kvasir': 0.918, 'CVC-ColonDB': 0.773,'ETIS-LaribPolypDB': 0.733, 'test':0.83}
        plt.axhline(y=transfuse[name[i]], color=color[i], linestyle='-')

    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.title('Train')
    plt.legend()
    plt.savefig('eval.png')
    # plt.show()



if __name__ == '__main__':
    dict_plot = {'CVC-300':[], 'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[]}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']

    ################## model_name #############################
    model_name = 'PolypPVT'
    ###############################################

    parser = argparse.ArgumentParser()

    #default=100
    parser.add_argument('--epoch', type=int,
                        default=2, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=True, help='choose to do random flip rotation')
    
    # default=8
    parser.add_argument('--batchsize', type=int,
                        default=3, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        #default='./dataset/TrainDataset/',
                        default='//NAS_home/Develop/Coding/Research/HSNet/dataset/TrainDataset/',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        #default='./dataset/TestDataset/',
                        default='//NAS_home/Develop/Coding/Research/HSNet/dataset/TestDataset/',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='./model_pth/'+model_name+'/')

    opt = parser.parse_args()
    logging.basicConfig(filename='train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = HSNet().cuda()

    best = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch):
         adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
         train(train_loader, model, optimizer, epoch, opt.test_path)

    #for epoch in range(1, opt.epoch):
    #
    #    if epoch in [15, 30]:
    #        adjust_lr(optimizer, 0.5)
    #
    #    train(train_loader, model, optimizer, epoch, opt.test_path)
    #
    # plot the eval.png in the training stage
    # plot_train(dict_plot, name)

    print("#" * 20, "  End Training  ", "#" * 20)

