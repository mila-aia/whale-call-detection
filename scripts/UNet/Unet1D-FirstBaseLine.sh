#!/bin/bash
  
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres gpu:rtx8000:1
#SBATCH --partition=unkillable
#SBATCH --output=/network/projects/aia//whale_call/LOGS/UNet1D-%j.out
#SBATCH --error=/network/projects/aia//whale_call/LOGS/UNet1D-%j.err
#SBATCH --mail-user=ge.li@mila.quebec
#SBATCH --mail-type=ALL

module --quiet load anaconda/3
conda activate whale

python whale/train.py fit --config experiments/unet/unet-1d.yaml
