#!/bin/bash
#SBATCH -J drugs_coords
#SBATCH --mail-user=tuan.le@pfizer.com
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu_medium
#SBATCH --constraint=weka
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 
#SBATCH --output=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/drugs_coords_%j.out
#SBATCH --error=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/drugs_coords_%j.err

# attempting to access the data directory
ls /hpfs/projects/mlcs/e3moldiffusion

cd /gpfs/workspace/users/let55/projects/e3moldiffusion/geom
source activate e3moldiffusion
echo "runnning experiment"

args=(
    --gpus 1
    --id 2
    --dataset drugs
    --max_num_conformers 30
    --num_workers 4
    --save_dir logs/drugs_coords
    --num_epochs 100
    --sdim 128 --vdim 32 --tdim 128 --edim 32 --rbf_dim 32 --num_layers 5
    --cutoff 10.0
    --lr 5e-4
    --batch_size 256
    # --local_global_model
    --fully_connected
    --use_bond_features
    --use_all_atom_features
    --omit_cross_product
    --vector_aggr mean
    --schedule cosine
    --beta_min 1e-4
    --beta_max 2e-2
    --num_diffusion_timesteps 300
    --max_time 00:23:45:00
    )

python train.py "${args[@]}"