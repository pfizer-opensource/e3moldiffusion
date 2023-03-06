#!/bin/bash
#SBATCH -J drugs_singleGPU
#SBATCH --mail-user=tuan.le@pfizer.com 
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --constraint=weka
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/drugs_singleGPU_%j.out
#SBATCH --error=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/drugs_singleGPU_%j.err

# attempting to access the data directory
ls /hpfs/projects/mlcs/e3moldiffusion

cd /gpfs/workspace/users/let55/projects/e3moldiffusion/geom
source activate e3moldiffusion
echo "runnning experiment"


args=(
    --gpus 1 --id 0
    --dataset drugs
    --max_num_conformers 30
    --num_workers 4
    --save_dir logs/drugs
    --num_epochs 100
    --sdim 64 --vdim 16 --tdim 64 --num_layers 5 
    --lr 5e-4 --batch_size 256
    --fully_connected 
    --use_bond_features
    --edim 16
    --use_all_atom_features
    --omit_cross_product
    --vector_aggr mean
    --schedule cosine
    --beta_min 1e-4
    --beta_max 2e-2
    --num_diffusion_timesteps 1000
    )

python train.py "${args[@]}"
