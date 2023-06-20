#!/bin/bash
#SBATCH -J large
#SBATCH --mail-user=tuan.le@pfizer.com
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --constraint=weka
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100:4
#SBATCH --output=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/new_run_%j.out
#SBATCH --error=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/new_run_%j.err

# attempting to access the data directory
ls /hpfs/projects/mlcs/e3moldiffusion

cd /gpfs/workspace/users/let55/projects/e3moldiffusion/experiments/geom
source activate e3moldiffusion
echo "runnning experiment"

args=(
    --gpus 4
    --id 11
    --dataset drugs
    --accum_batch 4
    --num_workers 4
    --save_dir logs/drugs_atomsbonds
    --num_epochs 100
    --sdim 256 --vdim 128 --rbf_dim 32 --edim 32 --num_layers 12
    --ema_decay 0.999
    --lr 4e-4
    --batch_size 44
    --fully_connected
    --omit_cross_product
    --vector_aggr mean
    --schedule cosine
    --num_diffusion_timesteps 500
    --max_time 03:23:45:00
    --test_interval 5
    )

python train_coordsatomsbonds_x0.py "${args[@]}"