#!/bin/bash -l

### https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster_advanced.html

#SBATCH -J drugs_experiment_multiGPU
#SBATCH --mail-user=tuan.le@pfizer.com 
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4 
#SBATCH --output=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/multiGPU_experiment_%j.out
#SBATCH --error=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/multiGPU_experiment_%j.err

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

cd /gpfs/workspace/users/let55/projects/e3moldiffusion/geom
source activate e3moldiffusion
echo "runnning experiment"

# todo: Check how it will work...
# srun python train.py --gpus 4 --id 1 --subset_frac 1.0 --num_workers 4 --energy_preserving --save_dir logs/diffusion --num_epochs 100