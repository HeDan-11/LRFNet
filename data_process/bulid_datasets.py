import os
from glob import glob
import cv2
import torch
import numpy as np
from data_process.Datasets import DatasetFromFolder_w as DatasetFromFolder
# from data_process.Datasets import DatasetFromFolder_w
# from Datasets import DatasetFromFolder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def get_list(root_dir, img_path):
    """
    路径已经被集成在.txt里面，给根目录和文件路径就可以直接读取
    遍历MRI_filenames即获得相应位置的图像路径
    """
    MRI_list, other_list = [], []
    f = open(img_path)
    lines = f.readlines()
    for line in lines:
        line1, line2 = line.strip().split(' ')
        MRI_list.append(root_dir + line1)
        other_list.append(root_dir + line2)
    return MRI_list, other_list


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp",".gif"])
# 针对一般论文中的情况，直接从文件夹获取，传入对应的路径
def get_list_2(MRI_PATH, OTHER_PATH):
    """
    train
    ---pair1 MRI
    ---pair2 OTHER(CT、PET、SPECT)
    """
    MRI_list = [os.join(MRI_PATH, x) for x in os.listdir(MRI_PATH) if is_image_file(x)]
    OTHER_list = [os.join(OTHER_PATH, x) for x in os.listdir(OTHER_PATH) if is_image_file(x)]
    """
    # 这个方式知道后缀名时使用比较方便
    MRI_list = glob(MRI_PATH + '/*.jpg')
    OTHER_list = glob(OTHER_PATH + '/*.jpg')
    """
    return MRI_list, OTHER_list


def get_train_set(root_dir, img_path, way='pair'):
    MRI_list, OTHER_list = get_list(root_dir, img_path)
    if way == 'seq':  # 第一种方式是依此输入
        train_set = DatasetFromFolder(MRI_list+OTHER_list, None, None)
    if way == 'pair':  # 第二种方式是成对输入
        # list1 = MRI_list+OTHER_list
        # list2 = OTHER_list+MRI_list
        # train_set = DatasetFromFolder(list1, list2, None)
        train_set = DatasetFromFolder(OTHER_list, MRI_list, None)
    return train_set


def load_test_list(data_root, MRI, OTHER):
    # data_root = f'E:/python-project/pytorch17/Medical_image_fusion/data/test_images/'
    MRI_PATH = data_root + f'{MRI}-{OTHER}/{MRI}'
    OTHER_PATH = data_root + f'{MRI}-{OTHER}/{OTHER}'
    MRI_list = [os.path.join(MRI_PATH, x) for x in os.listdir(MRI_PATH) if is_image_file(x)]
    Other_list = [os.path.join(OTHER_PATH, x) for x in os.listdir(OTHER_PATH) if is_image_file(x)]
    MRI_list.sort()
    Other_list.sort()

    return MRI_list, Other_list

def ycbcr_to_rgb(OTHER_img_y, OTHER_cb, OTHER_cr):
    OTHER_img_y = OTHER_img_y.data[0].numpy()
    OTHER_img_y *= 255.0
    OTHER_img_y = OTHER_img_y.clip(0, 255)
    all_img = np.stack([np.uint8(OTHER_img_y), OTHER_cb, OTHER_cr], axis=-1)
    result = cv2.cvtColor(all_img, cv2.COLOR_YCrCb2BGR)
    return result, np.uint8(OTHER_img_y)

def get_test_img(MRI_path, Other_path):
    trans = transforms.ToTensor()
    MRI_img = cv2.imread(MRI_path, cv2.IMREAD_GRAYSCALE)
    OTHER_img = cv2.imread(Other_path)
    rgb_to_yuv = cv2.cvtColor(OTHER_img, cv2.COLOR_BGR2YCrCb)
    OTHER_img_y, OTHER_cb, OTHER_cr = rgb_to_yuv[:, :, 0], rgb_to_yuv[:, :, 1], rgb_to_yuv[:, :, 2]
    MRI_img = trans(MRI_img)
    OTHER_img_y = trans(OTHER_img_y)
    return MRI_img, OTHER_img_y, OTHER_cb, OTHER_cr


if __name__ == '__main__':
    data_root = f'E:/python-project/pytorch17/Medical_image_fusion/data/test_images/'
    MRI, OTHER = 'MRI', 'PET'
    MRI_list, Other_list = load_test_list(data_root, MRI, OTHER)
    for i, (MRI_path, Other_path)in enumerate(zip(MRI_list, Other_list)):
        print(i)
        print(MRI_path)
        print(Other_path)
        MRI_img, OTHER_img_y, OTHER_cb, OTHER_cr = get_test_img(MRI_path, Other_path)
        # input_img = torch.cat((OTHER_img_y, MRI_img), 0).unsqueeze(0)

        print(OTHER_img_y.shape, OTHER_cb.shape)
        result = ycbcr_to_rgb(OTHER_img_y, OTHER_cb, OTHER_cr)
        print(result.shape)
        plt.imshow(result[:,:,::-1])
        plt.show()
        break
