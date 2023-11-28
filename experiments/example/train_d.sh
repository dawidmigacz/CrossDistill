#!/bin/bash
#SBATCH --job-name=job_name
#SBATCH --time=01:00:00
#SBATCH --account=plgcrossdistillphd-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu

CONFIG=$1

module load CUDA
python ../../tools/train_val.py --config ${CONFIG}