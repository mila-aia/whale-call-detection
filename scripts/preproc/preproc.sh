#!/bin/bash
  
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres gpu:rtx8000:1
#SBATCH --partition=unkillable
#SBATCH --output=/network/projects/aia/whale_call/LOGS/preprocess_labels_out.out
#SBATCH --error=/network/projects/aia/whale_call/LOGS/preprocess_labels_err.err
#SBATCH --mail-user=basile.roth@mila.quebec
#SBATCH --mail-type=ALL

module --quiet load anaconda/3
conda activate whale

python scripts/preprocess_labels.py 