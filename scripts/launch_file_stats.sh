#!/bin/bash
  
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=main
#SBATCH --output=/network/projects/aia/whale_call/LOGS/output_stats_data.log
#SBATCH --error=/network/projects/aia/whale_call/LOGS/error_stats_data.log


module --quiet load anaconda/3
conda activate whale

python scripts/SAC_file_data_analysis.py