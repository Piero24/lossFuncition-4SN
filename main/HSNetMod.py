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
    """
    folder_path = "./model_pth_200/PolypPVT/"
    """

    pth_files = glob.glob(folder_path + "*.pth")
    total_model = len(pth_files) 

    for index_pth, file_path in enumerate(pth_files):

        print(f"Start with the model ({index_pth}/{total_model}): {file_path}")
        model_name_splitted = file_path.split('/')[-1].split('.')[0]

        parser.add_argument('--pth_path', type=str, default = file_path)
        opt = parser.parse_args()
        model = HSNet()
        model.load_state_dict(torch.load(opt.pth_path))
        model.cuda()
        model.eval()
        for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:

            ##### put data_path here #####
            #data_path = './dataset/TestDataset/{}'.format(_data_name)
            data_path = '../dataset/TestDataset/{}'.format(_data_name)
            ##### save_path #####
            #save_path = './result_map/PolypPVT/{}/'.format(_data_name)
            save_path = '../result_map/PolypPVT/{}/{}/'.format(model_name_splitted, _data_name)

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            image_root = '{}/images/'.format(data_path)
            gt_root = '{}/masks/'.format(data_path)
            num1 = len(os.listdir(gt_root))
            test_loader = test_dataset(image_root, gt_root, 352)
            for i in range(num1):
                image, gt, name = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = image.cuda()
                P1,P2,P3,P4 = model(image)
                res = F.upsample(P1+P2+P3+P4, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                cv2.imwrite(save_path+name, res*255)

            print(_data_name, 'Finish!')
        
    print("#" * 20, "  End mask estraction  ", "#" * 20)
        


# if __name__ == '__main__':
#     hsnet_mask_writer("./model_pth_200/PolypPVT/")