#!/bin/bash -l

#SBATCH -J drugs_multiGPU
#SBATCH --mail-user=tuan.le@pfizer.com 
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu_medium
#SBATCH --constraint=weka
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:v100:4
#SBATCH --output=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/drugs_multiGPU_%j.out
#SBATCH --error=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/drugs_multiGPU_%j.err

# attempting to access the data directory
ls /hpfs/projects/mlcs/e3moldiffusion

cd /gpfs/workspace/users/let55/projects/e3moldiffusion/geom
source activate e3moldiffusion
echo "runnning multi-gpu experiment"


args=(
    --gpus 4 --id 1
    --dataset drugs
    --max_num_conformers 30
    --num_workers 4
    --save_dir logs/drugs
    --num_epochs 100
    --sdim 128 --vdim 32 --tdim 128 --num_layers 5 
    --lr 5e-4 --batch_size 256
    --fully_connected 
    --use_bond_features
    --edim 32
    --use_all_atom_features
    --omit_cross_product
    --vector_aggr mean
    --schedule cosine
    --beta_min 1e-4
    --beta_max 2e-2
    --num_diffusion_timesteps 300
    )


srun python train.py "${args[@]}"
