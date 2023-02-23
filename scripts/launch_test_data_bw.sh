#!/bin/bash
  
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=main
#SBATCH --output=/network/projects/aia/whale_call/LOGS/output_bw_data.log
#SBATCH --error=/network/projects/aia/whale_call/LOGS/error_bw_data.log


module --quiet load anaconda/3
conda activate whale

python scripts/bw_sac_data_quality.py