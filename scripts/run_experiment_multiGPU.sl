#!/bin/bash -l

### https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster_advanced.html

#SBATCH -J drugs_experiment_multiGPU
#SBATCH --mail-user=tuan.le@pfizer.com 
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --constraint=weka
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G

## https://github.com/Lightning-AI/lightning/issues/4612
#SBATCH --gres=gpu:4 
#SBATCH --output=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/multiGPU_experiment_%j.out
#SBATCH --error=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/multiGPU_experiment_%j.err

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

cd /gpfs/workspace/users/let55/projects/e3moldiffusion/geom
source activate e3moldiffusion
echo "runnning multi-gpu experiment"

srun python train.py --gpus 4 --id 3 --max_num_conformers 50 --num_workers 1 --save_dir logs/diffusion --num_epochs 100