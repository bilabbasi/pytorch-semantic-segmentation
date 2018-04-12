import os

import numpy as np
import scipy.io as sio
import torch
from PIL import Image, ImageOps
from torch.utils import data

num_classes = 21
ignore_label = 255
root = '/home/bilalabbasi/scratch/VOC/' # Compute canada server pwd
#root = '/Users/bilalabbasi/Dropbox/Projects/semantic-segmentation/VOC/'
'''
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''
palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def make_dataset(mode,set):
    assert mode in ['train', 'val', 'eval']
    assert set in ['benchmark','voc']
    items = []
    if mode == 'train':
        if set == 'benchmark':
            img_path = os.path.join(root, 'benchmark_RELEASE', 'dataset', 'img')
            mask_path = os.path.join(root, 'benchmark_RELEASE', 'dataset', 'cls')
            data_list = [l.strip('\n') for l in open(os.path.join(
                root, 'benchmark_RELEASE', 'dataset', 'train.txt')).readlines()]
            for it in data_list:
                item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.mat'))
                items.append(item)
        else:
            img_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'JPEGImages')
            mask_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'SegmentationClass')
            data_list = [l.strip('\n') for l in open(os.path.join(
                root, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation', 'train.txt')).readlines()]
            for it in data_list:
                item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
                items.append(item)
    elif mode == 'val':
        img_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'JPEGImages')
        mask_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'SegmentationClass')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation', 'seg11valid.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
            items.append(item)
    else:
        img_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'JPEGImages')
        mask_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'SegmentationClass')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation', 'seg11valid.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), it, os.path.join(mask_path, it + '.png'))
            items.append(item)
    return items


class VOC(data.Dataset):
    def __init__(self, mode,set='benchmark', joint_transform=None, transform=None, target_transform=None):
        self.imgs = make_dataset(mode,set)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.set = set
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        if self.mode == 'eval':
            img_path, img_name,mask_path = self.imgs[index]
            img = Image.open(img_path).convert('RGB')
            msk = Image.open(mask_path)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                msk = self.target_transform(msk)
            return img_name, img, msk

        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')

        if self.set == 'benchmark':
            assert self.mode == 'train', 'benchmark dataset can only be used for training' 
            mask = sio.loadmat(mask_path)['GTcls']['Segmentation'][0][0]
            mask = Image.fromarray(mask.astype(np.uint8))
        else:
            mask = Image.open(mask_path)

        Hnew,Wnew = 512,512
        dH = int(Hnew - img.size[0])
        dW = int(Wnew - img.size[1])
        if dH%2==0:
            dH1 = dH/2
            dH2 = dH/2
        else:
            dH1 = (dH-1)/2
            dH2 = dH1+1
        if dW%2==0:
            dW1 = dW/2
            dW2 = dW/2
        else:
            dW1 = (dW-1)/2
            dW2 = dW1+1
        padding = (dH1,dW1,dH2,dW2)
        img = ImageOps.expand(img,padding)
        mask = ImageOps.expand(mask,padding)
        
        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)

        if self.transform is not None:
                img = self.transform(img)
        if self.target_transform is not None:
                mask = self.target_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.imgs)
