import os
import time
import argparse
from glob import glob
import numpy as np
import cv2
import torch
from data_process.bulid_datasets import get_test_img, ycbcr_to_rgb
from nets.LRFNet import mymodel as net


device = torch.device('cuda:0')
use_gpu = torch.cuda.is_available()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--status', type=str, default='train', help='train or test')
    parser.add_argument('--model_name', type=str, default='model_name', help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float)
    parser.add_argument('--weight', default=[1, 1,0.0005, 0.00056], type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--alpha', default=300, type=int, help='number of new channel increases per depth (default: 300)')
    parser.add_argument('--checkpoint', type=str, default='./models', help='train or test')
    args = parser.parse_args()
    return args

def run_test():
    print("=========Starting building a test framework==========")
    # Saving path of fusion result
    SAVE_ROOT = './results/'
    # Saving path of test data
    data_root = './data'
    modalitys = ['PET',  'SPECT']
    model = net()
    # The path of the model
    model_dir = './models/'
    if use_gpu:
        model = model.cuda()
        model.load_state_dict(torch.load(model_dir + opt.checkpoint))
    else:
        state_dict = torch.load(model_dir + opt.checkpoint, map_location='cpu')
        model.load_state_dict(state_dict)

    for OTHER in modalitys:
        SAVE_PATH = SAVE_ROOT + f'MRI-{OTHER}/'
        MRI_list = glob(data_root + f'/MRI-{OTHER}/MRI/*.jpg')  # MRI/01.jpg 02.jpg 03.jpg
        Other_list = glob(data_root + f'/MRI-{OTHER}/{OTHER}/*.jpg')  # PET+SPECT/01.jpg 02.jpg 03.jpg
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        tic1 = time.time()
        for index, (MRI_path, Other_path) in enumerate(zip(MRI_list, Other_list)):
            # tic = time.time()
            MRI_img_c, OTHER_img_y_c, OTHER_cb_c, OTHER_cr_c = get_test_img(MRI_path, Other_path)
            input_img_c = torch.cat((OTHER_img_y_c, MRI_img_c), 0).unsqueeze(0)
            if use_gpu:
                input_img = input_img_c.cuda()
                OTHER_img_y = OTHER_img_y_c.unsqueeze(0).cuda()
                MRI_img = MRI_img_c.unsqueeze(0).cuda()

            tic = time.time()
            out_y = model(MRI_img, OTHER_img_y)
            print(f"test time: {time.time()-tic}")
            out_y = out_y.squeeze(0).cpu()
            # Convert colors and write locally.
            result, fuse_img_y = ycbcr_to_rgb(out_y, OTHER_cb_c, OTHER_cr_c)
            if index+1 < 10 :
                index1 = "0%s" % (index+1)
            else:
                index1 = str(index+1)

            cv2.imwrite(SAVE_PATH + f'{index1}.jpg', result)


def feature(x,save_path):
    # x --- torch.Size([1, 64, 256, 256])
    feature1 = x[0, :, :, :].cpu()
    for i in range(feature1.shape[0]):
        feature2 = feature1.data[i].numpy()
        pmin = np.min(feature2)
        pmax = np.max(feature2)
        feature2 = (feature2 - pmin) / (pmax - pmin + 0.000001)
        feature2 = np.uint8(feature2 * 255.0)

        cv2.imwrite(save_path+f"{i}.jpg", feature2)


if __name__ == '__main__':
    localtime = time.strftime("%Y%m%d_%H%M", time.localtime())  # 20230816_2351
    opt = get_args()
    opt.status = 'test'
    opt.checkpoint = f'model.pth'
    run_test()
