import sys
#sys.path.insert(0, '/Users/bilalabbasi/Dropbox/Projects/semantic-segmentation/pytorch-semantic-segmentation') # local root
sys.path.insert(0,'/home/babbasi/level-set-rnn') # compute canada root

import datetime
import os
import random
import torch as th
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.cuda as cuda
import torchvision.transforms as standard_transforms

import utils.transforms as extended_transforms
import models.fcn8s as model
# import models.fcn16s as model
# import models.u_net as model
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
# log_dir = '/home/babbasi/level-sets/pytorch-semantic-segmentation/train/voc-fcn'
log_dir = './logs/voc-fcn' # local log directory
def main(train_args):
    net = model.FCN8s(num_classes=voc.num_classes,pretrained=False)
    # net = model.FCN16VGG(num_classes=voc.num_classes,pretrained=False)
    # net = model.UNet(num_classes=voc.num_classes)


    if th.cuda.is_available():
        net=net.cuda()

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

    if th.cuda.is_available():
        criterion = CrossEntropyLoss2d(size_average=False, ignore_index=voc.ignore_label).cuda()
    else:
        criterion = CrossEntropyLoss2d(size_average=False, ignore_index=voc.ignore_label)
    optimizer = optim.SGD(net.parameters(),lr=0.01)

    os.makedirs(log_dir,exist_ok=True)
    training_log = open(log_dir+'/store_data.txt', 'w+') # Will write file if it doesn't exist
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=train_args['lr_patience'], min_lr=1e-10, verbose=True)

    for epoch in range(curr_epoch, train_args['epoch_num'] + 1):
        print('epoch = {}\n'.format(curr_epoch))
        train(train_loader, net, criterion, optimizer, epoch, train_args,training_log)

    training_log.close()
    
def train(train_loader, net, criterion, optimizer, epoch, train_args,training_log):
    train_loss = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    i = 0
    for data in train_loader:
        inputs, labels = data
        assert inputs.size()[2:] == labels.size()[1:]
        N = inputs.size(0)

        if th.cuda.is_available():
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)

        optimizer.zero_grad()

        outputs = net(inputs)
        assert outputs.size()[2:] == labels.size()[1:]
        assert outputs.size()[1] == voc.num_classes

        loss = criterion(outputs, labels) / N

        loss.backward()

        optimizer.step()

        print(loss.data[0])
        train_loss.update(loss.data[0], N)
        training_log.write(str(loss.data[0]))
        curr_iter += 1
        i+=1

if __name__ == '__main__':
    main(args)
