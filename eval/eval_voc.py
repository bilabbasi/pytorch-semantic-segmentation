import os
import sys
sys.path.append('..')

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as standard_transforms
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader

import utils.transforms as extended_transforms
from datasets import voc
#from models import *
from models import fcn8s,deeplab_resnet,dl
from utils import check_mkdir, evaluate

import pdb

cudnn.benchmark = True

ckpt_path = '../logs/ckpt/voc-fcn8s'

args = {
    'exp_name': 'voc-psp_net',
    'snapshot': 'epoch_33_loss_0.31766_acc_0.92188_acc-cls_0.81110_mean-iu_0.70271_fwavacc_0.86757_lr_0.0023769346.pth'
}
root = '/home/bilalabbasi/projects/pytorch-semantic-segmentation/models/pretrained'
model = 'fcn'
#pth = 'MS_DeepLab_resnet_trained_VOC.pth'
pth = 'test.pth'
def main():
    net = fcn8s.FCN8s(num_classes=voc.num_classes,pretrained=False).cuda()
    #net = deeplab_resnet.Res_Deeplab().cuda()
    #net = dl.Res_Deeplab().cuda()
    
    #net.load_state_dict(torch.load(os.path.join(ckpt_path, args['exp_name'], args['snapshot'])))
    net.load_state_dict(torch.load(os.path.join(root,model,pth)))
    net.eval()

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    val_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = extended_transforms.MaskToTensor()
    test_set = voc.VOC('eval', transform=val_input_transform,target_transform=target_transform)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=8, shuffle=False)

    #check_mkdir(os.path.join(ckpt_path, args['exp_name'], 'test'))
    predictions = []
    masks = []
    ious =np.array([])
    for vi, data in enumerate(test_loader):
        img_name, img, msk = data
        img_name = img_name[0]
        
        H,W = img.size()[2:]
        L = min(H,W)
        interp_before = nn.UpsamplingBilinear2d(size=(L,L))
        interp_after = nn.UpsamplingBilinear2d(size=(H,W))
        
        img = Variable(img, volatile=True).cuda()
        msk = Variable(msk, volatile=True).cuda()
        masks.append(msk.data.squeeze_(0).cpu().numpy())
        
        #img = interp_before(img)
        output = net(img)
        #output = interp_after(output[3])

        prediction = output.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
        #prediction = output.data.max(1)[1].squeeze().cpu().numpy()
        ious = np.append(ious,get_iou(prediction,masks[-1]))
        
        predictions.append(prediction)
        ## prediction.save(os.path.join(ckpt_path, args['exp_name], 'test',
        img_name + '.png')) prediction = voc.colorize_mask(prediction)
        prediction.save(os.path.join(root,'segmented-images',model+'-'+img_name+'.png'))
        #if vi == 10:
        #    break
        print('%d / %d' % (vi + 1, len(test_loader)))
    results = evaluate(predictions,masks,voc.num_classes)
    print('mean iou = {}'.format(results[2]))
    print(ious.mean())

def get_iou(x,y,num_labels=21):
    # x = prediction, y = ground truth
    labels = np.unique(y)
    class_iou = np.zeros((len(labels),))
    for i in range(len(labels)):
        x_ind = (x==int(labels[i]))
        y_ind = (y==int(labels[i]))
        intersection = (x_ind & y_ind).astype(float).sum()
        union = (x_ind | y_ind).astype(float).sum()
        class_iou[i] =  intersection/union
    return class_iou.mean()


if __name__ == '__main__':
    main()
