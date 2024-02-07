#!/bin/bash
#SBATCH --job-name=job_name
#SBATCH --time=00:40:00
#SBATCH --account=plgcrossdistillphd-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu
#SBATCH -o ./out/slurm-%j.out # STDOUT



CONFIG=$1

module load CUDA/12.0.0
cp ../../data/KITTI/ImageSets/orig/val.txt ../../data/KITTI/ImageSets/val.txt
python ../../tools/train_val.py --config kitti_example_values.yaml -e


