#!/bin/bash

#SBATCH --job-name=call-level-class-reg-lstm-bw_HQ_filt_mixed
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres gpu:1
#SBATCH --partition=long
#SBATCH --output=/network/projects/aia//whale_call/LOGS/%x-%j.out
#SBATCH --error=/network/projects/aia//whale_call/LOGS/%x-%j.err
#SBATCH --mail-type=ALL

module --quiet load anaconda/3
conda activate whale
PROJECT="whale-call-detection"
EXP_NAME="call-level-class-reg-lstm-bw_HQ_filt_mixed"
DATA_PATH="/network/projects/aia/whale_call/LABELS/BWC_3CH_HQ"
python models/lstm-optim.py \
    --project $PROJECT \
    --exp-name $EXP_NAME \
    --save-dir /network/projects/aia/whale_call/wandb_log/$PROJECT/$EXP_NAME \
    --data-path $DATA_PATH \
    --num-classes 2 \
    --batch-size 64 \
    --input-dim 129 \
    --search-num-epochs 10 \
    --num-epochs 30 \
    --hparams-space ./hparams/call-level-class-reg-lstm.yaml \
    --metric-to-optimize overall_val_loss \
    --optimize-direction minimize \
    --n-trials 20 \
    --optuna-db /network/projects/aia/whale_call/optuna/$EXP_NAME.sqlite3