## https://github.com/baiboat/HSNet/blob/main/Test.py
import glob

import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.pvt import HSNet
from utils.dataloader import test_dataset
import cv2




def hsnet_mask_writer(folder_path: str) -> None:
    """Extracts masks using the HSNet model for a given folder path.

    Args:
        folder_path (str): The path to the folder containing the model files.

    Returns:
        None

    Note:
        folder_path = "./model_pth_200/PolypPVT/"
    """

    # Find all .pth files in the specified folder
    pth_files = glob.glob(folder_path + "*.pth")
    total_model = len(pth_files) 

    # Iterate over each .pth file
    for index_pth, file_path in enumerate(pth_files):

        print(f"Start with the model ({index_pth}/{total_model}): {file_path}")
        model_name_splitted = file_path.split('/')[-1].split('.')[0]

        # Add the '--pth_path' argument to the argument parser and parse 
        # the arguments
        parser.add_argument('--pth_path', type=str, default = file_path)
        opt = parser.parse_args()
        # Create an instance of the HSNet model
        model = HSNet()
        # Load the model's state dictionary from the specified .pth file
        model.load_state_dict(torch.load(opt.pth_path))
        # Move the model to the GPU
        model.cuda()
        # Set the model to evaluation mode
        model.eval()
        # Iterate over each data name
        for _data_name in ['CVC-300']:
        # for _data_name in ['CVC-300', 
        #                    'CVC-ClinicDB',
        #                    'Kvasir', 
        #                    'CVC-ColonDB', 
        #                    'ETIS-LaribPolypDB']:

            ##### put data_path here #####
            #data_path = './dataset/TestDataset/{}'.format(_data_name)
            data_path = './dataset/TestDataset/{}'.format(_data_name)
            ##### save_path #####
            #save_path = './result_map/PolypPVT/{}/'.format(_data_name)
            save_path = './result_map/PolypPVT/{}/{}/'.format(model_name_splitted, 
                                                               _data_name)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            
            image_root = '{}/images/'.format(data_path)
            gt_root = '{}/masks/'.format(data_path)

            # Get the number of ground truth files
            num1 = len(os.listdir(gt_root))
            # Create a test dataset loader
            test_loader = test_dataset(image_root, gt_root, 352)

            for i in range(num1):
                # Load the image, ground truth, and name from the test loader
                image, gt, name = test_loader.load_data()
                # Convert the ground truth to a numpy array and normalize it
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                # Move the image to the GPU
                image = image.cuda()
                # Pass the image through the model
                P1,P2,P3,P4 = model(image)
                # Upsample the predicted masks to the size of the ground truth
                res = F.upsample(P1+P2+P3+P4, 
                                 size=gt.shape, 
                                 mode='bilinear', 
                                 align_corners=False)
                
                # Sigmoid activation, convert to numpy array, and squeeze the 
                # dimensions
                res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # Save the resulting mask as an image
                cv2.imwrite(save_path+name, res*255)

            print(_data_name, 'Finish!')
        
    print("#" * 20, "  End mask estraction  ", "#" * 20)
        


# if __name__ == '__main__':
#     hsnet_mask_writer("./model_pth_200/PolypPVT/")