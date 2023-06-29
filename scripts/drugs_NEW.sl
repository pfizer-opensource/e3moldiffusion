#!/bin/bash
#SBATCH -J embDIFF
#SBATCH --mail-user=tuan.le@pfizer.com
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=25G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --nodelist=g001
#SBATCH --output=/hpfs/userws/let55/projects/e3moldiffusion/experiments/geom/slurm_outs/drugs_EMBDiff_%j.out
#SBATCH --error=/hpfs/userws/let55/projects/e3moldiffusion/experiments/geom/slurm_outs/drugs_EMBDiff_%j.err

# delta
# cd /gpfs/workspace/users/let55/projects/e3moldiffusion/experiments/geom

# alpha
cd /hpfs/userws/let55/projects/e3moldiffusion/experiments/geom


source activate e3moldiffusion
echo "runnning experiment"

args=(
    --gpus 4
    --id 5
    --dataset drugs
    # --accum_batch 2
    --num_workers 4
    --save_dir logs/x0
    --num_epochs 300
    --sdim 256 --vdim 128 --rbf_dim 32 --edim 32 --num_layers 12
    --ema_decay 0.999
    --lr 4e-4
    --batch_size 50
    --fully_connected
    --omit_cross_product
    --vector_aggr mean
    --schedule cosine
    --num_diffusion_timesteps 500
    --timesteps 500
    --max_time 03:23:45:00
    --test_interval 400
    # --load_ckpt /hpfs/userws/let55/projects/e3moldiffusion/experiments/geom/logs/x0/run4/epoch=32-step=151173.ckpt
    )

python train_coordsatomsbonds_x0_embDF.py "${args[@]}"
