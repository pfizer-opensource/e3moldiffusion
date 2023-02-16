#!/bin/bash -l

### https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster_advanced.html  <- doesn't work
### https://github.com/PrincetonUniversity/multi_gpu_training/tree/main/03_pytorch_lightning

#SBATCH -J drugs_experiment_multiGPU
#SBATCH --mail-user=tuan.le@pfizer.com 
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu_short
#SBATCH --constraint=weka
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:4 
#SBATCH --output=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/multiGPU_experiment_%j.out
#SBATCH --error=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/multiGPU_experiment_%j.err

# debugging flags (optional)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export PYTHONFAULTHANDLER=1

cd /gpfs/workspace/users/let55/projects/e3moldiffusion/geom
source activate e3moldiffusion
echo "runnning multi-gpu experiment"
export PL_TORCH_DISTRIBUTED_BACKEND=gloo

srun python train.py --gpus 4 --id 1 --max_num_conformers 30 --num_workers 4 --save_dir logs/diffusion --num_epochs 100 --lr 5e-4 --num_layers 4 --sdim 128 --vdim 32 --tdim 32