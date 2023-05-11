#!/bin/bash
#SBATCH -J bond_large
#SBATCH --mail-user=tuan.le@pfizer.com
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu_medium
#SBATCH --constraint=weka
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:2
#SBATCH --output=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/large_bond%j.out
#SBATCH --error=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/large_bond%j.err

# attempting to access the data directory
ls /hpfs/projects/mlcs/e3moldiffusion

cd /gpfs/workspace/users/let55/projects/e3moldiffusion/geom
source activate e3moldiffusion
echo "runnning experiment"

args=(
    --gpus 2
    --id 22
    --dataset drugs
    --num_workers 4
    --save_dir logs/drugs_atomsbonds
    --num_epochs 100
    --sdim 512 --vdim 256 --rbf_dim 32 --num_layers 8 
    --ema_decay 0.999
    --cutoff_local 7.0
    --lr 2e-4
    --batch_size 32
    --local_global_model
    # --fully_connected
    --omit_cross_product
    --vector_aggr mean
    --schedule cosine
    --num_diffusion_timesteps 1000
    --max_time 00:23:45:00
    )

python train_coordsatomsbonds.py "${args[@]}"