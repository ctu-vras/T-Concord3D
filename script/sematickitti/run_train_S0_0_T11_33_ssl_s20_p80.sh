#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=24                  # 1 node
#SBATCH --ntasks-per-node=1  # 36 tasks per node
#SBATCH --time=21-00:00:0              # time limits: 500 hour
#SBATCH --partition=amdgpuextralong	  # gpufast
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --output=logs/semantickitti_train_S0_0_T11_33_ssl_s20_p80_%j.log     # file name for stdout/stderr

ml spconv/2.1.21-foss-2021a-CUDA-11.3.1
ml PyTorch-Geometric/2.0.2-foss-2021a-CUDA-11.3.1-PyTorch-1.10.0

cd ../..

name=T-Concord3D

python train.py --config_path 'config/semantickitti/semantickitti_S0_0_T11_33_ssl_s20_p80.yaml' \
 2>&1 | tee logs_dir/${name}_logs_semantickitti_S0_0_T11_33.txt
