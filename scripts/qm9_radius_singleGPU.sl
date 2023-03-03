#!/bin/bash
#SBATCH -J qm9_coords_radius
#SBATCH --mail-user=tuan.le@pfizer.com 
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu_short
#SBATCH --constraint=weka
#SBATCH --time=0-01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 
#SBATCH --output=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/qm9_coords_radius_%j.out
#SBATCH --error=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/qm9_coords_radius_%j.err

# attempting to access the data directory
ls /hpfs/projects/mlcs/e3moldiffusion

cd /gpfs/workspace/users/let55/projects/e3moldiffusion/geom
conda activate e3moldiffusion
echo "runnning experiment"

args=(
    --gpus 1 --id 2
    --dataset qm9
    --max_num_conformers 30
    --num_workers 4
    --save_dir logs/qm9_coords
    --num_epochs 20
    --sdim 64 --vdim 16 --tdim 64 --num_layers 4 
    --lr 5e-4 --batch_size 256
    # --fully_connected 
    --local_global_model
    --use_bond_features
    --edim 16
    --cutoff 5.0
    --rbf_dim 16
    --use_all_atom_features
    --omit_cross_product
    --vector_aggr mean
    --schedule cosine
    --beta_min 1e-4
    --beta_max 2e-2
    --num_diffusion_timesteps 300
    )

python train.py "${args[@]}"
