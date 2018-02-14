# # !usr/bin/env python3
# This script generates a batch script to be run on the Compute Canada server
import argparse
import subprocess
import time, os
# import numpy as np
from parsers import parser
parser = argparse.ArgumentParser('Script for running code for image segmentation')
parser.add_argument('--cpus',type=int,default=2)
parser.add_argument('--gpus',type=int,default=2)
parser.add_argument('--m',type=int,default=64000)
parser.add_argument('---fname',type=str,default='test')

gpus = args.gpus
mem = args.mem
cpus = args.cpu
#log_dir = os.path.join('..','logs')
s = open('cc-runner.sh','w')

log_dir = '/home/level-set-rnn/logs'
# For Python 2
s.write('#!/bin/bash\n')
s.write('#SBATCH --account=def-oberman\n')
# s.write('#SBATCH --time='+time.strftime('%H:%M:%S',t)+' \t\t# max time (HH:MM:SS)\n')
s.write('#SBATCH --mem='+str(mem)+'M \t\t\t# memory per node\n')
s.write('#SBATCH --cpus-per-task='+str(cpus) +'\n')
s.write('#SBATCH --output='+log_dir+'/log.out\n')
s.write('#SBATCH --gpus=gpu:'+str(gpus)+'\n')
s.write('#SBATCH --signal=15@30 \t\t#Send SIGTERM 30 seconds before time out\n')
s.write('\n\nsource ~/anaconda3/bin/activate\n')
s.write('python -u /home/babbasi/level-set-rnn/train/voc-fcn/train.py')
s.close()

# For Python 3
# print('#!/bin/bash\n',file=s)
# print('#SBATCH --account=def-oberman\n',file=s)
# # s.write('#SBATCH --time='+time.strftime('%H:%M:%S',t)+' \t\t# max time (HH:MM:SS)\n',file=s)
# print('#SBATCH --job-name='+'test\n',fgile=s)
# print('#SBATCH --mem='+str(mem)+'M \t\t\t# memory per node\n',file=s)
# print('#SBATCH --cpus-per-task='+str(cpus) +'\n',file=s)
# # s.write('#SBATCH --output='+log_dir+'/log.out\n')
# print('#SBATCH --signal=15@30 \t\t#Send SIGTERM 30 seconds before time out\n',file=s)
# print('\n\nsource ~/anaconda3/bin/activate\n',file=s)
# print('python -u /home/babbasi/level-set/pytorch-image-segmentation/train/voc-fcn/train-test.py',file=s,flush=True)
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

subprocess.call(['sbatch' ,'cc-runner.sh'])
