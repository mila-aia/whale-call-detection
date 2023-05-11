#!/bin/bash

#SBATCH --job-name=baselines_lstm_call_level_class_reg
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres gpu:1
#SBATCH --partition=long
#SBATCH --output=/network/projects/aia//whale_call/LOGS/%x-%j.out
#SBATCH --error=/network/projects/aia//whale_call/LOGS/%x-%j.err
#SBATCH --mail-type=ALL

module --quiet load anaconda/3
conda activate whale

python whale/main.py fit --config experiments/call-level-class-reg/lstm-fwc-1ch-hq.yaml
