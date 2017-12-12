import sys
# sys.path.insert(0, '/Users/bilalabbasi/Dropbox/Projects/net-lsm/pytorch-semantic-segmentation/') # cpu root
sys.path.insert(0, '/home/babbasi/level-sets/pytorch-semantic-segmentation/') # compute canada root

import datetime
import os
import random

from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.cuda as cuda
import torchvision.transforms as standard_transforms

import utils.transforms as extended_transforms
from models import *
from datasets import voc
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d

cudnn.benchmark = True

args = {
    'epoch_num': 1,
    'lr': 1e-10,
    'weight_decay': 1e-4,
    'momentum': 0.95,
    'lr_patience': 100,  # large patience denotes fixed lr
    'snapshot': '',  # empty string denotes learning from scratch
    'print_freq': 20,
    'val_save_to_img_file': False,
    'val_img_sample_rate': 0.1  # randomly sample some validation results to display
}

log_dir = '/home/babbasi/level-sets/pytorch-semantic-segmentation/train/voc-fcn'
# model = models.fcn8s()
def main(train_args):
    net = models.FCN8s(num_classes=voc.num_classes,pretrained=False).cuda()

    curr_epoch = 1

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

    criterion = CrossEntropyLoss2d(size_average=False, ignore_index=voc.ignore_label).cuda()

    optimizer = optim.SGD(net.params,lr=0.1)

    os.makedirs(log_dir + '/store_data.csv',exist_ok=True)
    training_log = open(log_dir, 'w')
    for epoch in range(curr_epoch, train_args['epoch_num'] + 1):
        train(train_loader, net, criterion, optimizer, epoch, train_args,training_log)

    training_log.close()
    
def train(train_loader, net, criterion, optimizer, epoch, train_args,training_log):
    train_loss = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    for i, data in enumerate(train_loader):
        inputs, labels = data
        N = inputs.size(0)

        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, labels) / N

        loss.backward()

        optimizer.step()

        train_loss.update(loss.data[0], N)
        training_log.write(loss.data[0] + '\n')
        curr_iter += 1


if __name__ == '__main__':
    main(args)