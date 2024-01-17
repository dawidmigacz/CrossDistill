#!/bin/bash
#SBATCH --job-name=job_name
#SBATCH --time=15:40:00
#SBATCH --account=plgcrossdistillphd-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu
#SBATCH -o ./out/slurm-%j.out # STDOUT



CONFIG=$1
module load CUDA/12.0.0
cp ../../data/KITTI/ImageSets/r93.txt ../../data/KITTI/ImageSets/val.txt
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
cp ../../data/KITTI/ImageSets/d93.txt ../../data/KITTI/ImageSets/val.txt
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/
python ../../tools/train_val.py --config kitti_example_centernet_rgb.yaml -e
rm -rf rgb_outputs/data/