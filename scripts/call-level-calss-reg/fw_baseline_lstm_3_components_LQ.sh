#!/bin/bash

#SBATCH --job-name=baselines_lstm_call_level_class_reg_fwc_3CH_LQ
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres gpu:rtx8000:1
#SBATCH --partition=long
#SBATCH --output=/network/projects/aia//whale_call/LOGS/%x-%j.out
#SBATCH --error=/network/projects/aia//whale_call/LOGS/%x-%j.err
#SBATCH --mail-type=ALL

module --quiet load anaconda/3
conda activate whale

PROJECT="whale-call-detection"
EXP_NAME="FW-3-comp-call-level-class-reg"
RUN_NAME="LQ"

python models/lstm.py \
    --data-path /network/projects/aia/whale_call/LABELS/FWC_3CH_LQ \
    --save-dir /network/projects/aia/whale_call/exp_results \
    --project $PROJECT \
    --exp-name $EXP_NAME \
    --run-name $RUN_NAME \
    --input-dim 129 \
    --hidden-dim 128 \
    --num-layers 3 \
    --lr 0.001 \
    --num-classes 2 \
    --batch-size 64 \
    --num-epochs 30 \
    --data-type 'spec' \
    --metric-to-optimize 'overall_val_loss' 