#!/bin/bash

#SBATCH --job-name=test-comet-logger
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
EXP_NAME="test-comet-logger"
RUN_NAME="HQ"

python models/lstm.py \
    --data-path /network/projects/aia/whale_call/LABELS/BWC_3CH_HQ \
    --save-dir /network/projects/aia/whale_call/comet_log/$PROJECT/$EXP_NAME \
    --project $PROJECT \
    --exp-name $EXP_NAME \
    --run-name $RUN_NAME \
    --input-dim 129 \
    --hidden-dim 64 \
    --num-layers 3 \
    --lr 0.001 \
    --num-classes 2 \
    --batch-size 128 \
    --num-epochs 2 \
    --data-type 'spec' \
    --metric-to-optimize 'overall_val_loss' 