#!/bin/bash
  
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres gpu:1
#SBATCH --partition=main
#SBATCH --output=/network/projects/aia/whale_call/LOGS/UNet1D-mini-%j.out
#SBATCH --error=/network/projects/aia/whale_call/LOGS/UNet1D-mini-%j.err
#SBATCH --mail-user=ge.li@mila.quebec
#SBATCH --mail-type=ALL

module --quiet load anaconda/3
conda activate whale

python whale/main.py fit --config experiments/unet/unet-1d-mini.yaml
