import argparse
import subprocess
import time, os
import numpy as np

parser = argparse.ArgumentParser('Script for running BGD tests on Compute Canada')
parser.add_argument('--dataset', type=str, default='mnist', metavar='DS',
                    choices=['mnist','cifar10'],
                    help='dataset. Either "cifar10" or "mnist". (default: mnist)')
args = parser.parse_args()

ccscript = open('ccrunner.sh','w')
gpus = 4

print('#!/bin/bash',file=)
print('#SBATCH --account=def-oberman',file=ccscript)
print('#SBATCH --time='+time.strftime('%H:%M:%S',t)+' \t\t# max time (HH:MM:SS)',
        file=ccscript)
print('#SBATCH --job-name='+args.algorithm+'_'+args.dataset,
        file=ccscript)
print('#SBATCH --mem='+str(mem)+'M \t\t\t# memory per node', file=ccscript)
if not args.no_cuda:
    print('#SBATCH --gres=gpu:'+str(gpus)+' \t\t# request 4 gpus per node', 
        file=ccscript)
print('#SBATCH --cpus-per-task='+str(cpus), file=ccscript)
print('#SBATCH --output='+log_dir+'/log.out', file=ccscript)
print('#SBATCH --signal=15@30 \t\t#Send SIGTERM 30 seconds before time out', file=ccscript)

print('\n\nsource ~/anaconda3/bin/activate', file=ccscript)


print('python -u ~/optimization/bgd/main.py  --data ~/scratch'+ '\\\n'
        '\t--momentum ' + str(args.momentum) + '\\\n'
        '\t--lipshitz ' + str(args.lipshitz)+'\\\n'+
        '\t--lipshitz-schedule "' + str(lsched)+'"\\\n'+
        '\t--log-dir '  +log_dir+'\\\n'+
        '\t--epochs '  +str(args.epochs)+'\\\n'+
        '\t--dataset '  +args.dataset+'\\\n'+
        '\t--algorithm ' +args.algorithm+'\\\n'+
        '\t--gamma '+str(args.gamma0)+'\\\n'+
        '\t--weight-decay '+str(args.weight_decay)+'\\\n'+
        '\t--gamma-schedule "'+str(gsched)+'"\\\n'+
        '\t--buf-decay '+str(args.buf_decay)+'\\\n'+
        '\t--max-inner '+str(args.max_inner)+'\\\n'+
        '\t--min-inner '+str(args.min_inner)+'\\\n'+
        keep_mom_str+
        no_cuda_str+
        '\t--model '+args.model
    ,file=ccscript, flush=True)

subprocess.call(['sbatch' ,'ccrunner.sh'])