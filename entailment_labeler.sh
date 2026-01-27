#!/bin/bash
#SBATCH --job-name=entailment_labeler
#SBATCH --output=log/entailment_labeler.out
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=64G
#SBATCH --chdir=/home/edenzha/flow_of_argument

echo "=== Debug Info ==="
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -c "import torch; print('torch.cuda.device_count():', torch.cuda.device_count())"
echo "=================="


python data_processing/entailment_labeler.py --k 512
