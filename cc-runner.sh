#!/bin/bash
#SBATCH --account=def-oberman
#SBATCH --job-name=test
#SBATCH --mem=16000M 			# memory per node
#SBATCH --cpus-per-task=16
#SBATCH --output=../logs/log.out
#SBATCH --signal=15@30 		#Send SIGTERM 30 seconds before time out


source ~/anaconda3/bin/activate
python -u ~/level-sets/pytorch-image-segmentation/train/voc-fcn/train-test.py