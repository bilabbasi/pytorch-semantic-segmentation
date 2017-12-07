# # !/usr/bin/env python3
# This script generates a batch script to be run on the Compute Canada server
import argparse
import subprocess
import time, os
# import numpy as np

# parser = argparse.ArgumentParser('Script for running code for image segmentation')
# parser.add_argument('--dataset', type=str, default='mnist', metavar='DS',
#                     choices=['mnist','cifar10'],
#                     help='dataset. Either "cifar10" or "mnist". (default: mnist)')
# parser.add_argument('-model', type=str,default='fcn16s')

# args = parser.parse_args()

s = open('cc-runner.sh','w')
gpus = 4
mem = 16000
cpus = 16
log_dir = os.path.join('..','logs')
s.write('#!/bin/bash\n')
s.write('#SBATCH --account=def-oberman\n')
# s.write('#SBATCH --time='+time.strftime('%H:%M:%S',t)+' \t\t# max time (HH:MM:SS)\n')
s.write('#SBATCH --job-name='+'test\n')
s.write('#SBATCH --mem='+str(mem)+'M \t\t\t# memory per node\n')
# if not args.no_cuda:
#     s.write('#SBATCH --gres=gpu:'+str(gpus)+' \t\t# request 4 gpus per node\n')
s.write('#SBATCH --cpus-per-task='+str(cpus) +'\n')
# s.write('#SBATCH --output='+log_dir+'/log.out\n')
# s.write('#SBATCH --output='+'~/level-set/pytorch-image-segmentation/train/voc-fcn/train-test.py' + '\n')
s.write('#SBATCH --signal=15@30 \t\t#Send SIGTERM 30 seconds before time out\n')
s.write('\n\nsource ~/anaconda3/bin/activate\n')
s.write('python -u ~/level-set/pytorch-image-segmentation/train/voc-fcn/train-test.py')
# print('python -u ~/level-set/pytorch-image-segmentation/train/voc-fcn/train.py' ,file=ccscript, flush=True)
# print('python -u ~/optimization/bgd/main.py  --data ~/scratch'+ '\\\n'
#         '\t--momentum ' + str(args.momentum) + '\\\n'
#         '\t--lipshitz ' + str(args.lipshitz)+'\\\n'+
#         '\t--lipshitz-schedule "' + str(lsched)+'"\\\n'+
#         '\t--log-dir '  +log_dir+'\\\n'+
#         '\t--epochs '  +str(args.epochs)+'\\\n'+
#         '\t--dataset '  +args.dataset+'\\\n'+
#         '\t--algorithm ' +args.algorithm+'\\\n'+
#         '\t--gamma '+str(args.gamma0)+'\\\n'+
#         '\t--weight-decay '+str(args.weight_decay)+'\\\n'+
#         '\t--gamma-schedule "'+str(gsched)+'"\\\n'+
#         '\t--buf-decay '+str(args.buf_decay)+'\\\n'+
#         '\t--max-inner '+str(args.max_inner)+'\\\n'+
#         '\t--min-inner '+str(args.min_inner)+'\\\n'+
#         keep_mom_str+
#         no_cuda_str+
#         '\t--model '+args.model
#     ,file=ccscript, flush=True)
s.close()

subprocess.call(['sbatch' ,'cc-runner.sh'])