#!/bin/bash
#SBATCH -J EQGAT_X0
#SBATCH --mail-user=tuan.le@pfizer.com
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --time=6-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=27G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --nodelist=g002
#SBATCH --output=/hpfs/userws/let55/projects/e3moldiffusion/experiments/geom/slurm_outs/drugs_EMBDiff_%j.out
#SBATCH --error=/hpfs/userws/let55/projects/e3moldiffusion/experiments/geom/slurm_outs/drugs_EMBDiff_%j.err

# delta
# cd /gpfs/workspace/users/let55/projects/e3moldiffusion/experiments/geom

# alpha
cd /hpfs/userws/let55/projects/e3moldiffusion/experiments/geom


source activate e3moldiffusion_new
echo "runnning experiment"

args=(
    --gpus 4
    --id 11
    --dataset drugs
    --accum_batch 1
    --grad_clip_val 10.0
    --num_workers 4
    --save_dir logs/x0
    --num_epochs 300
    --sdim 256 --vdim 128 --rbf_dim 32 --edim 96 --num_layers 12
    # --ema_decay 0.999
    --gamma 0.99
    --lr 4e-4   # 2e-4
    --batch_size 45
    --fully_connected
    --omit_cross_product
    --vector_aggr mean
    --schedule cosine
    --num_diffusion_timesteps 500
    --timesteps 500
    --max_time 05:23:45:00
    --test_interval 10
    #--load_ckpt /hpfs/userws/let55/projects/e3moldiffusion/experiments/geom/logs/x0/run10/epoch=20-step=126383.ckpt
    --save_dir /hpfs/userws/let55/projects/e3moldiffusion/experiments/geom/logs/x0
    )

python train_coordsatomsbonds_x0.py "${args[@]}"
