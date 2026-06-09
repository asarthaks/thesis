#!/bin/bash -l

#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=2
#SBATCH --job-name=c_p_on_m
#SBATCH --output=slurm-logs/output.%N.%j.log
#SBATCH --error=slurm-logs/output.%N.%j.log
#SBATCH --gres=gpu:2
#SBATCH --nodelist=destc0strapp19
## SBATCH --gres=gpu:1,gpu_mem:30000

source ~/projects/decoding/.venv/bin/activate
 
echo $VIRTUAL_ENV

python -c "import torch

# Check if CUDA is available
print(torch.cuda.is_available()) # Should return True if CUDA is available

# Attempt to allocate a tensor on the GPU
try:
    torch.tensor([1.0, 2.0]).cuda()
except RuntimeError as e:
    print(e) # RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or unavailable"

cd ~/projects/decoding/controlled_text_generation/
python Experiments/IdealAlphaSchedule/GradientInformation/evaluate_cls_multi.py