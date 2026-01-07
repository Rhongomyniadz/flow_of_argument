#!/bin/bash
#SBATCH --job-name=llm_as_a_judge
#SBATCH --output=llm_as_a_judge.out
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:2
#SBATCH --mem=64G
#SBATCH --chdir=/home/edenzha/flow_of_argument

export CUDA_HOME=/usr/local/cuda-12.6
export PATH=/usr/local/cuda-12.6/bin:/opt/anaconda/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64
export TORCH_CUDA_ARCH_LIST="8.0"
export FLASHINFER_COMPUTE_CAPS=80

python llm_as_a_judge.py \
  --input analysis_charts/clarification_prediction/questions.csv \
  --output analysis_charts/clarification_prediction/questions_labeled.csv \
  --batch_size 64
