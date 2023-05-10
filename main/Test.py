# https://github.com/baiboat/HSNet/blob/main/Test.py

import os
# Parses the command-line arguments passed to the program.
import argparse

import cv2 
import torch
import numpy as np
from scipy import misc
# Provides useful functions for building neural networks, 
# such as activation functions and loss functions
import torch.nn.functional as F

from lib.pvt import HSNet
# Functions to get the data loaders, while
from utils.dataloader import test_dataset

# Contains folder paths or other constants useful in code
import folder_path


if __name__ == '__main__':

    # Created an ArgumentParser object from argparse to handle command line arguments
    parser = argparse.ArgumentParser()

    # Specifies the size of the test dataset, i.e. 
    # the total number of training samples
    parser.add_argument('--testsize', type=int, 
                        default=352, help='testing size')
    
    # parser.add_argument('--pth_path', type=str, 
    #                    default='./model_pth/HSNet.pth')

    # Specifies the path to the model's state dictionary file.
    # This path is used to load the trained model
    parser.add_argument('--pth_path', type=str, 
                        default='./model_pth/PolypPVT/PolypPVT.pth')
    
    # Parse the command-line arguments provided when running the script
    opt = parser.parse_args()
    # The HSNet model is instantiated using the default constructor 
    model = HSNet()
    # It loads the model's state dictionary from the file
    model.load_state_dict(torch.load(opt.pth_path))
    # Moved to the GPU using the .cuda() method
    model.cuda()
    # Sets the model to evaluation mode 
    # to disable certain operations like dropout or 
    # batch normalization during testing
    model.eval()
    
    for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:

        ##### put data_path here #####
        # 
        # Sets the variable to the path of the current dataset based on the dataset name
        data_path = folder_path.MY_TEST_FOLDER_PATH + '/{}'.format(_data_name)
        ##### save_path #####
        # 
        # Sets the variable to the path where the result maps will be 
        # saved for the current dataset
        save_path = './result_map/PolypPVT/{}/'.format(_data_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Sets the path of the images and mask folder 
        # for the current dataset
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        # Retrieves the number of samples in the dataset
        num1 = len(os.listdir(gt_root))
        # Create an object using the test_dataset class, providing the image and 
        # ground truth paths and the testing size
        test_loader = test_dataset(image_root, gt_root, 352)

        # Iterates over the number of samples in the dataset
        for i in range(num1):
            # Loads a test sample
            image, gt, name = test_loader.load_data()
            # Converts the ground truth to a NumPy array 
            # and normalizes it
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            # Moves the image tensor to the GPU
            image = image.cuda()
            # INFO: What does it do?
            P1, P2, P3, P4 = model(image)
            # Resizes the combined probability maps to the size of the 
            # ground truth using bilinear interpolation
            res = F.interpolate(P1+P2+P3+P4, size=gt.shape, mode='bilinear', align_corners=False)
            # Applies the sigmoid function to the resized 
            # probability maps and converts the result to a NumPy array
            res = res.sigmoid().data.cpu().numpy().squeeze()
            # Performs min-max normalization
            #res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            # Saves the result map as an image
            cv2.imwrite(save_path+name, res*255)

        print(_data_name, 'Finish!')
