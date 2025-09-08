#!/bin/bash
#SBATCH --job-name=assumption_extraction
#SBATCH --output=slurm-%j.out
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:A6000:1         # (uncomment only if your site uses typed GRES)
#SBATCH --mem=32G
#SBATCH --chdir=/home/edenzha/flow_of_argument

# --- Load site toolchains (adapt versions/names to your cluster) ---
module purge
module load cuda/12.1

# --- Activate your conda/venv that has CUDA-enabled PyTorch + vLLM ---
source ~/.bashrc

# --- Quick visibility checks go to the job log ---
echo "=== nvidia-smi ==="
nvidia-smi || { echo "No GPU visible; aborting."; exit 1; }

echo "=== torch CUDA sanity ==="
python - <<'PY'
import torch, os
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY


# --- Run ---
python extract_assumption.py \
  --data_path  /home/edenzha/flow_of_argument/data/covid_episodes.jsonl.gz \
  --turns_path /home/edenzha/flow_of_argument/data/covid_episodes_turn.jsonl.gz