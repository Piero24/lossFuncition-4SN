import os
import logging
# Parses the command-line arguments passed to the program
import argparse
from datetime import datetime

import torch
import numpy as np
# Allows you to create tensors with support for 
# automatic gradient calculation
from torch.autograd import Variable
# Provides useful functions for building neural networks, 
# such as activation functions and loss functions
import torch.nn.functional as F
import matplotlib.pyplot as plt

from lib.pvt import HSNet
# Functions to get the data loaders, while
from utils.dataloader import get_loader, test_dataset
# Utility functions or classes for handling gradients, 
# adjusting the learning rate, and calculating averages
from utils.utils import clip_gradient, adjust_lr, AvgMeter

# Loss Function
from loss import bce_loss, dice_loss, IoU_loss
from loss import dice_bce_loss, log_cosh_dice_loss
from loss import focal_loss, tversky_loss
from loss import focal_tversky_loss, combo_loss
from loss import structure_loss
from lossTest import RWLoss

# Contains folder paths or other constants useful in code
import folder_path


def test(model: torch.nn.Module, path: str, dataset: str) -> float:
    """Compute the Dice Similarity Coefficient (DSC) on a dataset using a given model.

    Args:
        model: the PyTorch model to use for testing.
        path (str): the path to the directory containing the dataset.
        dataset (str): the name of the dataset to test.

    Returns:
        float: the average DSC value computed on the dataset.
    
    WARNING:
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
    
    Notes:
        The Dice Similarity Coefficient (DSC) is a metric commonly used to evaluate the 
        similarity between two sets, in particular it is often used to measure the quality 
        of a segmentation in computer vision problems, such as the segmentation of medical images.

        DSC measures the overlap between two sets using the following formula:

        DSC = (2 * |A ∩ B|) / (|A| + |B|)

        where |A| represents the dimension of the set A and |B| 
        represents the dimension of the set B. |A ∩ B| represents the size 
        of the intersection between A and B, i.e. the number of elements common 
        to the two sets.

        In the context of medical image segmentation, set A corresponds to 
        the segmentation mask generated by the algorithm or model, 
        while set B corresponds to the reference segmentation mask, 
        which represents the ground truth. 
        The DSC then measures the similarity between the segmentation 
        produced and the reference segmentation.

        The DSC value varies from 0 to 1, where 0 indicates a complete 
        lack of overlap between the two segmentations and 1 indicates 
        a perfect overlap.

        In the code you provided, the DSC is calculated for each image 
        in the testing stage. 
        The segmentation results are first interpolated (or upsample) 
        to fit the size of the reference mask. Next, the DSC is calculated 
        using the formula described above.

        Finally, the average of the DSC values obtained on all the images 
        of the test dataset is calculated and returned as a result of 
        the `test()` function. This average value represents the overall 
        performance of the model on the segmentation of the test dataset.



    """

    # Build paths to images and masks
    #data_path = os.path.join(path, dataset)
    data_path = '{}/{}'.format(path, dataset)
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
        # Load test data from test_loader
        image, gt, name = test_loader.load_data()
        # Convert gt to a numpy array of type float32
        gt = np.asarray(gt, np.float32)
        # Normalize gt by dividing by its maximum value, 
        # so that it has values between 0 and 1
        gt /= (gt.max() + 1e-8)
        # Move image to GPU
        image = image.cuda()
        
        # Performs image segmentation using the model model. 
        # Returns the segmentation predictions corresponding 
        # to res, res1, res2, and res3
        res, res1, res2, res3  = model(image)

        # eval Dice
        # Upsample result and compute sigmoid
        #
        # Performs a bilinear interpolation of the summed segmentation 
        # predictions (res, res1, res2, and res3) to match the size of 
        # the segmentation labels (gt)
        res = F.interpolate(res + res1 + res2 + res3 , size=gt.shape, mode='bilinear', align_corners=False)
        # Applies the sigmoid function to the res segmentation predictions, 
        # then converts the result into a numpy array, trimming off any 
        # unnecessary dimensions
        res = res.sigmoid().data.cpu().numpy().squeeze()
        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        # Compute Dice Similarity Coefficient
        input = res
        # Create a numpy target array from the gt variable 
        # containing the segmentation labels
        target = np.array(gt)
        # Assign the size of gt to variable N
        N = gt.shape
        # Defines a smoothing value to use when 
        # calculating DSC to avoid division by zero
        smooth = 1
        # Flattens the input array into a one-dimensional vector
        input_flat = np.reshape(input, (-1))
        # Flattens the target array into a one-dimensional vector
        target_flat = np.reshape(target, (-1))
        # Calculate the element-by-element intersection between input_flat and target_flat
        intersection = (input_flat * target_flat)
        # Calculate Dice's coefficient of similarity (DSC) using the formula 
        # (2 * intersection + smooth) / (input sum + target sum + smooth). 
        # The smooth value is added to both the numerator and denominator to 
        # avoid division by zero and ensure numerical stability
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        # Format the value of dice as a string with 4 decimal places
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        # Adds the value of dice to the DSC variable, 
        # which represents the cumulative sum of the similarity 
        # coefficients of dice calculated for all iterations
        DSC = DSC + dice
        # Compute average DSC and return it
        avg_DSC = DSC / num1
    
    return avg_DSC



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

    # Set model to train mode
    #
    # This affects the behavior of some components of the model, 
    # such as dropout regularization, which are activated during 
    # training but deactivated during the evaluation phase
    model.train()

    # Best score achieved during model training
    global best

    # initialize size rates for data rescaling
    #
    # Contains scaling factors used to scale images and training masks. 
    # These scaling factors are applied to generate different input sizes 
    # during training, allowing the model to learn at different scales
    size_rates = [0.75, 1, 1.25]

    # Initialize loss record
    #
    # Accumulator used to record the average value of 
    # the loss during model training
    loss_P2_record = AvgMeter()

    # iterate over training data batches
    # 
    # Double nested loop to iterate over both training 
    # data batches and scaling factors
    for i, pack in enumerate(train_loader, start=1):

        # Iterate over size rates for data rescaling
        for rate in size_rates:
            # Zero the gradients in the optimizer.
            # Reset the gradients accumulated in the previous step.
            optimizer.zero_grad()

            # ---- data prepare ----
            #
            # Batch data, images, and ground truth masks are extracted, 
            # which are then converted into Variable tensors and 
            # pushed to the GPU
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()

            # ---- rescale ----
            #
            # Scaling of images and masks, if the scaling rate factor 
            # is not equal to 1. 
            # Bilinear interpolation is used to fit images and 
            # masks to the size specified by trainsize.
            trainsize = int(round(opt.trainsize * rate / 32) * 32)

            # If the rate scaling factor is not equal to 1, 
            # images and masks are scaled to the size specified 
            # by trainsize using bilinear interpolation.
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            

            # Copy of the original images for later use
            # new_images = torch.clone(images)

            # ---- forward ----
            #
            # INFO: What does it do?
            P1, P2, P3, P4 = model(images)

            # Outputs are concatenated along dimension 1 to 
            # obtain a single combined_tensor tensor that 
            # combines information from different levels of depth
            combined_tensor = torch.cat((P1, P2, P3, P4), dim=1)

            # ---- loss function ----
            #
            # The losses for each of the four outputs are 
            # calculated using the structure_loss loss function, 
            # which compares the model predictions with 
            # the ground truth masks
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss_P3 = structure_loss(P3, gts)
            loss_P4 = structure_loss(P4, gts)
            loss = loss_P1 + loss_P2 + loss_P3 + loss_P4
            # Instantiating the RWLoss class
            rw_loss = RWLoss()
            # Calculated the loss
            loss = rw_loss.forward(combined_tensor, gts)
            # Calculated the loss
            # loss = rw_loss.forward(new_images, gts)

            # ---- backward ----
            # 
            # Backward pass by calling the backward() 
            # method on the loss. 
            # This computes the gradients of the model 
            # parameters with respect to the loss
            loss.backward()
            # The gradients are then capped 
            # to prevent them from getting too large
            clip_gradient(optimizer, opt.clip)
            # Update of model parameters. 
            # This updates the model parameters based on the gradients calculated during back propagation.
            optimizer.step()

            
            # ---- recording loss ----
            if rate == 1:
                # Object is updated with the current loss value
                # INFO: Why only with loss_P4?
                # loss_P2_record.update(loss_P4.data, opt.batchsize)
                # Object is updated with the current loss value
                loss_P2_record.update(loss.data, opt.batchsize) 
        
        # ---- train visualization ----
        #
        # It checks if the current iteration i is 
        # a multiple of 20 or if it's the last step
        if i % 20 == 0 or i == total_step:
            # Print the training progress
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}] lr'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_P2_record.show()), optimizer.param_groups[0]['lr'])

    # Save model
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Saves the model's state dictionary to a file in the save_path directory, 
    # with the filename consisting of the current epoch and the model name 
    # ('PolypPVT.pth')
    torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypPVT.pth')

    # ---- Choose the best model ----
    # 
    # Responsible for evaluating the model on test 
    # datasets and selecting the best model based 
    # on the evaluation results

    # Store the evaluation results
    global dict_plot

    # Path to the test dataset
    test1path = folder_path.MY_TEST_FOLDER_PATH

    # For every epoch
    if (epoch + 1) % 1 == 0:

        # Iterates over different datasets
        for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
            # Function to evaluate the model on that dataset 
            dataset_dice = test(model, test1path, dataset)
            # It logs the evaluation result, including the current epoch, dataset name, and dice score
            logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
            print(dataset, ': ', dataset_dice)
            # Appends the dice score of the dataset to the dict_plot 
            # dictionary for plotting purposes
            dict_plot[dataset].append(dataset_dice)
        
        # After evaluating on individual datasets, it 
        # evaluates the model on the 'test' dataset
        meandice = test(model, test_path, 'test')
        # Appends the dice score of the 'test' dataset 
        # to the dict_plot dictionary
        dict_plot['test'].append(meandice)

        # If the dice score of the 'test' dataset is 
        # higher than the current best score
        if meandice > best:
            # Updates the best score to the new dice score
            best = meandice
            # Saves the model's state dictionary 
            # to two separate files: 'PolypPVT.pth' and 'PolypPVT-best.pth
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

    # List of colors that will be used for the lines of the graphs
    color = ['red', 'lawngreen', 'lime', 'gold', 'm', 'plum', 'blue']
    line = ['-', "--"]

    # iterate over each element in the list 
    for i in range(len(name)):
        # Plot method is called to plot a graph using the data corresponding to dict_plot[name[i]]. 
        # The graph label is set as name[i] and the color and line style are taken 
        # from the color and line lists respectively
        plt.plot(dict_plot[name[i]], label=name[i], color=color[i], linestyle=line[(i + 1) % 2])
        # Contains some reference values associated with certain dataset names
        transfuse = {'CVC-300': 0.902, 'CVC-ClinicDB': 0.918, 'Kvasir': 0.918, 'CVC-ColonDB': 0.773,'ETIS-LaribPolypDB': 0.733, 'test':0.83}
        # Draw a horizontal line in the graph at a height specified by the value corresponding 
        # to the dataset name in the transfuse dictionary
        plt.axhline(y=transfuse[name[i]], color=color[i], linestyle='-')
    
    # Plot and show the graph
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.title('Train')
    plt.legend()
    plt.savefig('eval.png')
    plt.show()



if __name__ == '__main__':
    """
        WARNING: The batchsize and trainsize parameters are related. 
            The trainsize parameter must be a multiple of bach size. 
            For example if trainsize is 352 then batchsize can be set 
            to 2 to 4 or 8 but for example it cannot be 6 or 9. 
            This otherwise generates random errors in the code. 
            Unfortunately this is a basic AdamW bug that cannot be fixed.
    """
    # Empty dictionary that will be used to store the training curves of the different datasets. 
    # Dictionary keys are dataset names and values are empty lists that will be populated 
    # with training curves
    dict_plot = {'CVC-300':[], 'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[]}
    # List of dataset names corresponding to dictionary keys
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']

    ################## model_name #############################
    model_name = 'PolypPVT'
    ###############################################

    # Created an ArgumentParser object from argparse to handle command line arguments
    parser = argparse.ArgumentParser()

    # Specifies the number of epochs for model training
    parser.add_argument('--epoch', type=int,
                        # default=100
                        default=5, help='epoch number')
    
    # Specifies the learning rate used by the optimizer during training
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    
    # Specifies the optimizer to use during training.
    # You can choose between "AdamW" and "SGD"
    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')
    
    # Specifies whether or not to augment data during training. 
    # If set to True, random transformations such as rotation 
    # and flip will be applied to the training images
    parser.add_argument('--augmentation',
                        default=True, help='choose to do random flip rotation')
    
    # Specifies the training batch size, that is, the number 
    # of samples to use in a single training iteration
    parser.add_argument('--batchsize', type=int,
                        # default=8
                        default=2, help='training batch size')
    
    # Specifies the size of the training dataset, i.e. 
    # the total number of training samples
    parser.add_argument('--trainsize', type=int,
                        # default=352
                        default=352, help='training dataset size')
    
    # Specifies the limit value for clip gradients during training. 
    # If the absolute value of a gradient exceeds this limit, 
    # it will be reduced to this limit value
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    
    # Specifies the learning rate decay rate during training
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    
    # Specify after how many epochs to reduce the learning rate. 
    # At every multiple of decay_epoch, the learning rate is decreased 
    # using the specified decay rate
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    # Specifies the path to the training dataset. 
    # It is a string argument that contains the path to the training dataset
    parser.add_argument('--train_path', type=str,
                        #default='./dataset/TrainDataset/',
                        default=folder_path.MY_TRAIN_FOLDER_PATH,
                        help='path to train dataset')

    # Specifies the path to the test dataset. 
    # It is a string argument that contains the path to the test dataset
    parser.add_argument('--test_path', type=str,
                        #default='./dataset/TestDataset/',
                        default=folder_path.MY_TEST_FOLDER_PATH,
                        help='path to testing Kvasir dataset')

    # Specifies the path to save model files during training
    parser.add_argument('--train_save', type=str,
                        default='./model_pth/'+model_name+'/')
    
    # Command line arguments are parsed and the opt object is created to store them
    opt = parser.parse_args()
    # Record log messages
    logging.basicConfig(filename='train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device

    # The HSNet model is instantiated using the default constructor 
    # and then moved to the GPU using the .cuda() method
    model = HSNet().cuda()

    # Used to keep track of the best score achieved during model training
    best = 0

    # Used as input to the optimizer
    params = model.parameters()

    # The optimizer is created based on the choice specified 
    # in the optimizer argument passed via argparse.
    # If opt.optimizer is set to 'AdamW', the AdamW optimizer is used 
    # with the parameters params, opt.lr learning rate and 1e-4 decay weight.
    # Otherwise, the SGD optimizer is used with the parameters params, 
    # learning rate opt.lr, decay weight 1e-4 and momentum 0.9
    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    # Object created using the get_loader function, which loads training 
    # data from the specified folders and returns a dataloader
    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    # Is set to the length of the dataloader, which represents the total 
    # number of batches in the training
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch):
        # Adjusts the learning rate of the optimizer based on the current epoch. 
        # The function takes as arguments the optimizer (optimizer), initial 
        # learning rate (opt.lr), current epoch (epoch), 
        # learning rate reduction factor (0.1), and rate reduction 
        # epoch of learning (200)
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        # Train the model. 
        # Parameters include the training dataloader (train_loader), 
        # model (model), optimizer (optimizer), current epoch (epoch), 
        # and test data path (opt.test_path)
        train(train_loader, model, optimizer, epoch, opt.test_path)

    # for epoch in range(1, opt.epoch):
    #     # If the current epoch is equal to 15 or 30, 
    #     # # the adjust_lr function is called to reduce 
    #     # # the learning rate of the optimizer to 0.5
    #     if epoch in [15, 30]:
    #         adjust_lr(optimizer, 0.5)
        
    #     # Call the train function again to train the model with 
    #     # the current optimizer and parameters
    #     train(train_loader, model, optimizer, epoch, opt.test_path)
    
    # Plot the eval.png in the training stage
    plot_train(dict_plot, name)

    print("#" * 20, "  End Training  ", "#" * 20)

