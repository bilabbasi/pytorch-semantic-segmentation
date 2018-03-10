import os
import sys
sys.path.append('..')

import torch
import torchvision.transforms as standard_transforms
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader

import utils.transforms as extended_transforms
from datasets import voc
#from models import *
from models import fcn8s,deeplab_resnet
from utils import check_mkdir, evaluate

cudnn.benchmark = True

ckpt_path = '../logs/ckpt/voc-fcn8s'

args = {
    'exp_name': 'voc-psp_net',
    'snapshot': 'epoch_33_loss_0.31766_acc_0.92188_acc-cls_0.81110_mean-iu_0.70271_fwavacc_0.86757_lr_0.0023769346.pth'
}
root = '/home/bilalabbasi/projects/pytorch-semantic-segmentation/models/pretrained'
model = 'fcn'
pth = 'test.pth'
def main():
    net = fcn8s.FCN8s(num_classes=voc.num_classes,pretrained=False).cuda()
    #net = deeplab_resnet.Res_Deeplab().cuda()
    #print('load model ' + args['snapshot'])
    
    #net.load_state_dict(torch.load(os.path.join(ckpt_path, args['exp_name'], args['snapshot'])))
    net.load_state_dict(torch.load(os.path.join(root,model,pth)))
    net.eval()

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    val_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = extended_transforms.MaskToTensor()
    test_set = voc.VOC('test', transform=val_input_transform,target_transform=target_transform)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=8, shuffle=False)

    #check_mkdir(os.path.join(ckpt_path, args['exp_name'], 'test'))
    predictions = []
    masks = []
    for vi, data in enumerate(test_loader):
        img_name, img, msk = data
        img_name = img_name[0]

        img = Variable(img, volatile=True).cuda()
        msk = Variable(msk, volatile=True).cuda()
        masks.append(msk.data.squeeze_(0).cpu().numpy())
        
        output = net(img)
        prediction = output.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
        predictions.append(prediction)
         
        prediction = voc.colorize_mask(prediction)
        ## prediction.save(os.path.join(ckpt_path, args['exp_name], 'test', img_name + '.png'))
        prediction.save(os.path.join(root,'segmented-images',model+'-'+img_name+'.png'))

        print('%d / %d' % (vi + 1, len(test_loader)))
    results = evaluate(predictions,masks,voc.num_classes)
    print(results[2])

if __name__ == '__main__':
    main()
