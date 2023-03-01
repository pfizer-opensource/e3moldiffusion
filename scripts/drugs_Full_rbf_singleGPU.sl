#!/bin/bash
#SBATCH -J drugs_full_rbf
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
#SBATCH --output=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/drugs_full_rbf_%j.out
#SBATCH --error=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/drugs_full_rbf_%j.err

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
    --num_epochs 500
    --sdim 128 
    --rbf_dim 32
    --cutoff 10.0
    --vdim 32
    --tdim 64
    --num_layers 7 
    --lr 5e-4 
    --batch_size 256
    # --fully_connected 
    --edim 0
    --omit_cross_product
    --vector_aggr mean
    --schedule cosine
    --beta_min 1e-4
    --beta_max 2e-2
    --num_diffusion_timesteps 300
    )

python train_full.py "${args[@]}"

