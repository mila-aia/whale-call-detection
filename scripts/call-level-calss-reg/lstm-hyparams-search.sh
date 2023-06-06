#!/bin/bash

#SBATCH --job-name=call-level-class-reg-lstm-fw_HQ_filt_mixed
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres gpu:1
#SBATCH --partition=long
#SBATCH --output=/network/projects/aia//whale_call/LOGS/%x-%j.out
#SBATCH --error=/network/projects/aia//whale_call/LOGS/%x-%j.err
#SBATCH --mail-type=ALL

module --quiet load anaconda/3
conda activate whale
EXP_NAME="call-level-class-reg-lstm-fw_HQ_filt_mixed"
DATA_PATH="/network/projects/aia/whale_call/LABELS/fw_HQ_filt_mixed"
python models/lstm-optim.py \
    --exp-name $EXP_NAME \
    --mlruns-dir /network/projects/aia/whale_call/mlruns/ \
    --data-path $DATA_PATH \
    --num-classes 2 \
    --batch-size 64 \
    --input-dim 129 \
    --search-num-epochs 15 \
    --num-epochs 30 \
    --hparams-space ./hparams/call-level-class-reg-lstm.yaml \
    --metric-to-optimize overall_val_loss \
    --optimize-direction minimize \
    --n-trials 20 \
    --optuna-db /network/projects/aia/whale_call/optuna/$EXP_NAME.sqlite3