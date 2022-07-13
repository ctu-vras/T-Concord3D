#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=6                  # 1 node
#SBATCH --ntasks-per-node=1         # 36 tasks per node
#SBATCH --time=72:00:00               # time limits: 500 hour
#SBATCH --partition=amdgpulong	  # gpufast
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --output=logs/run_train_T2_2_s20_%j.log     # file name for stdout/stderr

ml spconv/2.1.21-foss-2021a-CUDA-11.3.1
ml PyTorch-Geometric/2.0.2-foss-2021a-CUDA-11.3.1-PyTorch-1.10.0

cd ../..
name=T-Concord3D

python train_T-Concord3d.py --config_path 'config/semantickitti/semantickitti_T2_2_s20.yaml' \
 2>&1 | tee logs_dir/${name}_logs_tee_T2_2_s20.txt
