#!/bin/bash
#SBATCH --account=def-oberman
#SBATCH --time=02:00:00
#SBATCH --job-name=test
#SBATCH --mem=64000M 			# memory per node
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/babbasi/level-set-rnn/logs/log.out
#SBATCH --signal=15@30 		#Send SIGTERM 30 seconds before time out

source ~/anaconda3/bin/activate
python -u /home/babbasi/level-set-rnn/train/voc-fcn/train.py
