#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=12                 # 1 node
#SBATCH --ntasks-per-node=1         # 36 tasks per node
#SBATCH --time=4:00:00               # time limits: 500 hour
#SBATCH --partition=amdgpufast	  # gpufast
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --output=logs/run_test_S0_0_T11:33_s20_p80_%j.log  # file name for stdout/stderr

ml spconv/2.1.21-foss-2021a-CUDA-11.3.1
ml PyTorch-Geometric/2.0.2-foss-2021a-CUDA-11.3.1-PyTorch-1.10.0

cd ../..
name=T-Concord3D

python3 test_T-Concord3d.py --config_path 'config/semantickitti/semantickitti_S0_0_T11_33_ssl_33_s20_p80.yaml' \
--mode 'test' --save 'True' --challenge 'True' 2>&1 | tee logs_dir/${name}_logs_test_f0_0_T11_33_ssl_s20_p80.txt
