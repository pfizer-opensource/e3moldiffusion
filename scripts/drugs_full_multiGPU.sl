#!/bin/bash -l

#SBATCH -J drugs_full_multiGPU
#SBATCH --mail-user=tuan.le@pfizer.com 
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --constraint=weka
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:v100:4
#SBATCH --output=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/drugs_full_multiGPU_%j.out
#SBATCH --error=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/drugs_full_multiGPU_%j.err

# attempting to access the data directory
ls /hpfs/projects/mlcs/e3moldiffusion

cd /gpfs/workspace/users/let55/projects/e3moldiffusion/geom
source activate e3moldiffusion
echo "runnning multi-gpu experiment"

args=(
    --gpus 4
    --id 11
    --dataset drugs
    --max_num_conformers 30
    --num_workers 4
    --save_dir logs/drugs
    --num_epochs 100
    --sdim 128 --vdim 32 --tdim 64 --edim 0 --rbf_dim 32 --num_layers 6
    --cutoff 10.0
    --lr 5e-4
    --batch_size 256
    --local_global_model
    #--use_bond_features
    #--use_all_atom_features
    --omit_cross_product
    --vector_aggr mean
    --schedule cosine
    --beta_min 1e-4
    --beta_max 2e-2
    --num_diffusion_timesteps 300
    --max_time 01:23:45:00
    )

srun python train_full.py "${args[@]}"
