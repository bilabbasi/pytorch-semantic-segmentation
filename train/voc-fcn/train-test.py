import sys
# sys.path.insert(0, '/Users/bilalabbasi/Dropbox/Projects/net-lsm/pytorch-semantic-segmentation/') # cpu root
sys.path.insert(0, '~/level-sets/pytorch-semantic-segmentation/') # compute canada root


import datetime
import os
import random

import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.cuda as cuda

import utils.transforms as extended_transforms
from datasets import voc
from models import *
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d

cudnn.benchmark = True

args = {
    'epoch_num': 300,
    'lr': 1e-10,
    'weight_decay': 1e-4,
    'momentum': 0.95,
    'lr_patience': 100,  # large patience denotes fixed lr
    'snapshot': '',  # empty string denotes learning from scratch
    'print_freq': 20,
    'val_save_to_img_file': False,
    'val_img_sample_rate': 0.1  # randomly sample some validation results to display
}


def main(train_args):
    net = FCN8s(num_classes=voc.num_classes,pretrained=False).cuda()

    curr_epoch = 1
    train_args['best_record'] = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}

    net.train()

    # Data loader and transformations
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = extended_transforms.MaskToTensor()
    restore_transform = standard_transforms.Compose([
        extended_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage(),
    ])
    visualize = standard_transforms.Compose([
        standard_transforms.Scale(400),
        standard_transforms.CenterCrop(400),
        standard_transforms.ToTensor()
    ])

    train_set = voc.VOC('train', transform=input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=1, num_workers=4, shuffle=True)
    val_set = voc.VOC('val', transform=input_transform, target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=4, shuffle=False)

    criterion = CrossEntropyLoss2d(size_average=False, ignore_index=voc.ignore_label).cuda()

    optimizer = optim.SGD(net.params,lr=0.1)

    for epoch in range(curr_epoch, train_args['epoch_num'] + 1):
        train(train_loader, net, criterion, optimizer, epoch, train_args)


def train(train_loader, net, criterion, optimizer, epoch, train_args):
    train_loss = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    for i, data in enumerate(train_loader):
        inputs, labels = data
        assert inputs.size()[2:] == labels.size()[1:]
        N = inputs.size(0)

        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, labels) / N
        loss.backward()
        optimizer.step()

        train_loss.update(loss.data[0], N)

        curr_iter += 1


if __name__ == '__main__':
    main(args)