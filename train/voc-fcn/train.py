import datetime, os, random, sys
import pdb
root = '/home/bilalabbasi/projects/pytorch-semantic-segmentation/'
sys.path.insert(0, root) # compute canada root

import torch
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
# from models import *
from models import fcn8s, fcn16s, fcn32s, deeplab_resnet, MBO
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d

cudnn.benchmark = True

ckpt_path = os.path.join(root,'logs','ckpt')
exp_name = 'voc-fcn8s'

args = {
    'epoch_num': 300,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'momentum': 0.95,
    'lr_patience': 100,  # large patience denotes fixed lr
    'snapshot': '',  # empty string denotes learning from scratch
    'print_freq': 20,
    'val_save_to_img_file': False,
    'val_img_sample_rate': 0.1  # randomly sample some validation results to display
}

model = 'fcn8s'
iter_freq = 50
epoch_freq = 20 # Frequency to save parameter states
bsz = 10
def main(train_args):
    if cuda.is_available():
        net = fcn8s.FCN8s(num_classes=voc.num_classes, pretrained=False).cuda() 
        #net = MBO.MBO().cuda()
        #net = deeplab_resnet.Res_Deeplab().cuda()
    else:
        print('cuda is not available')
        net = fcn8s.FCN8s(num_classes=voc.num_classes,pretrained=True)

    net.train()

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
    
    train_set = voc.VOC('train',set='benchmark', transform=input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=bsz, num_workers=8, shuffle=True)
    
    val_set = voc.VOC('val',set='voc', transform=input_transform, target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=4, shuffle=False)

    criterion = CrossEntropyLoss2d(size_average=False, ignore_index=voc.ignore_label).cuda()
    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': train_args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': train_args['lr']}],
        betas=(train_args['momentum'], 0.999))
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, min_lr=1e-10, verbose=True)

    lr0 = 1e-7
    max_epoch = 50
    max_iter = max_epoch * len(train_loader)
    #optimizer = optim.SGD(net.parameters(),lr = lr0, momentum = 0.9, weight_decay = 0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.5)

    log_dir = os.path.join(root,'logs','voc-fcn')
    time = datetime.datetime.now().strftime('%d-%m-%H-%M')
    train_file = 'train_log' + time + '.txt'
    val_file = 'val_log' + time + '.txt'
    #os.makedirs(log_dir,exist_ok=True) 
     
    training_log = open(os.path.join(log_dir,train_file),'w') 
    val_log = open(os.path.join(log_dir,val_file),'w')

    curr_epoch = 1
    for epoch in range(curr_epoch, train_args['epoch_num'] + 1):
        train(train_loader, net, criterion, optimizer, epoch, train_args,training_log, max_iter,lr0) 
        val_loss = validate(val_loader, net, criterion, optimizer, epoch, train_args, restore_transform, visualize,val_log)

        scheduler.step(val_loss) 
        
        lr_tmp = 0.0
        k = 0
        for param_group in optimizer.param_groups:
            lr_tmp += param_group['lr']
            k+=1
        val_log.write('learning rate = {}'.format(str(lr_tmp/k)) + '\n')
        #scheduler.step()

def train(train_loader, net, criterion, optimizer, epoch, train_args,training_log,max_iter,lr0):
    train_loss = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    for i, data in enumerate(train_loader):
        inputs, labels = data
        bsz = len(inputs)
        #pdb.set_trace()
        if cuda.is_available():
           inputs = Variable(inputs).cuda()
           labels = Variable(labels).cuda()
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)

        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, labels) / bsz
        loss.backward()
        optimizer.step()

        curr_iter += 1
        #poly_lr_step(optimizer,lr0,curr_iter,max_iter,power=0.9)
        train_loss.update(loss.data[0], bsz)
        
        training_log.write(str(curr_iter) + ' ' + str(train_loss.avg) +'\n')
        if curr_iter%iter_freq==0:
            print('epoch={}, it={} '.format(epoch,curr_iter),str(train_loss.avg))


def validate(val_loader, net, criterion, optimizer,  epoch, train_args, restore, visualize,val_log):
    net.eval()

    val_loss = AverageMeter()
    inputs_all, gts_all, predictions_all = [], [], []

    for vi, data in enumerate(val_loader):
        inputs, gts = data
        N = inputs.size(0)
        
        if cuda.is_available():
            inputs = Variable(inputs, volatile=True).cuda()
            gts = Variable(gts, volatile=True).cuda()
        else:
            inputs = Variable(inputs,volatile=True)
            gts = Variable(gts, volatile=True)

        outputs = net(inputs)
       
        loss = criterion(outputs,gts)/N
        val_loss.update(loss.data[0], N)
        
        #val_log.write(str(epoch) + ' ' + str(val_loss.avg) + '\n')

        inputs_all.append(inputs.data.squeeze_(0).cpu())
        gts_all.append(gts.data.squeeze_(0).cpu().numpy())
       
        predictions = outputs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
        predictions_all.append(predictions)
        
    acc, acc_cls, mean_iu, fwavacc = evaluate(predictions_all, gts_all, voc.num_classes)
    print('Mean IoU for epoch {} is {}'.format(epoch,mean_iu))
    val_log.write('epoch {}, average val loss = {}'.format(epoch,val_loss.avg))
    val_log.write('Mean IoU for epoch {} is {}'.format(epoch,mean_iu) + '\n')
    root = '/home/bilalabbasi/projects/pytorch-semantic-segmentation/logs/pths'
    if epoch%20 == 0:
        torch.save(net.state_dict(), os.path.join(root,model + '_epoch_' +str(epoch)+ '_iou_' + str(mean_iu)+ '.pth'))
        
    net.train()
    return val_loss.avg

def poly_lr_step(optimizer,lr0,iter,max_iter,power=0.9):
    lr = lr0 * (1-float(iter)/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
if __name__ == '__main__':
    main(args)
