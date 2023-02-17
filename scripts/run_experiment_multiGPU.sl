#!/bin/bash -l

### https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster_advanced.html  <- doesn't work
### https://github.com/PrincetonUniversity/multi_gpu_training/tree/main/03_pytorch_lightning

#SBATCH -J drugs_experiment_multiGPU
#SBATCH --mail-user=tuan.le@pfizer.com 
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu_medium
#SBATCH --constraint=weka
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:4
#SBATCH --output=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/multiGPU_experiment_discrete_%j.out
#SBATCH --error=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/multiGPU_experiment_discrete_%j.err

# debugging flags (optional)
# https://pytorch.org/docs/stable/distributed.html
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export PYTHONFAULTHANDLER=1
# export PL_TORCH_DISTRIBUTED_BACKEND=gloo  <-- slowly.

cd /gpfs/workspace/users/let55/projects/e3moldiffusion/geom
source activate e3moldiffusion
echo "runnning multi-gpu experiment"

srun python train.py --gpus 4 --id 2 --max_num_conformers 30 --num_workers 8 --save_dir logs/drugs --num_epochs 100 --lr 5e-4 --num_layers 5 --sdim 64 --vdim 16 --tdim 16 --batch_size 256 --fully_connected --omit_norm
