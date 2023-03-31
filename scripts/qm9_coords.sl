#!/bin/bash
#SBATCH -J qm9_coords
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
#SBATCH --output=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/qm9_coords_%j.out
#SBATCH --error=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/qm9_coords_%j.err

# attempting to access the data directory
ls /hpfs/projects/mlcs/e3moldiffusion

cd /gpfs/workspace/users/let55/projects/e3moldiffusion/geom
source activate e3moldiffusion
echo "runnning experiment"

args=(
    --gpus 1
    --id 0
    --dataset qm9
    --max_num_conformers 30
    --num_workers 4
    --save_dir logs/qm9_coords
    --num_epochs 100
    --sdim 64 --vdim 16 --tdim 64 --edim 16 --rbf_dim 16 --num_layers 4 
    --cutoff_local 3.0
    --cutoff_global 7.0
    # --conservative
    --lr 5e-4
    --batch_size 256
    --local_global_model
    --use_bond_features
    --use_all_atom_features
    --omit_cross_product
    --vector_aggr mean
    --schedule cosine
    --num_diffusion_timesteps 100
    --max_time 00:23:50:00
    )

python train.py "${args[@]}"
