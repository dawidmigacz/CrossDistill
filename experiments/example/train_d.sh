#!/bin/bash
#SBATCH --job-name=job_name
#SBATCH --time=01:40:00
#SBATCH --account=plgcrossdistillphd-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu
#SBATCH -o ./out/slurm-%j.out # STDOUT



CONFIG=$1

module load CUDA/12.0.0
python ../../tools/train_val.py --config kitti_example_centernet.yaml 