#!/bin/bash
#SBATCH -J qm9_full_rbf
#SBATCH --mail-user=tuan.le@pfizer.com
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu_medium
#SBATCH --constraint=weka
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 
#SBATCH --output=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/qm9_full_rbf_%j.out
#SBATCH --error=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/qm9_full_rbf_%j.err

# attempting to access the data directory
ls /hpfs/projects/mlcs/e3moldiffusion

cd /gpfs/workspace/users/let55/projects/e3moldiffusion/geom
source activate e3moldiffusion
echo "runnning experiment"


args=(
    --gpus 1 --id 2
    --dataset qm9
    --max_num_conformers 30
    --num_workers 4
    --save_dir logs/qm9
    --num_epochs 100
    --sdim 64 
    --rbf_dim 16
    --cutoff 7.5
    --vdim 16
    --tdim 64
    --num_layers 4
    --lr 5e-4 
    --batch_size 256
    # --fully_connected 
    --local_global_model
    --edim 0
    --omit_cross_product
    --vector_aggr mean
    --schedule cosine
    --beta_min 1e-4
    --beta_max 2e-2
    --num_diffusion_timesteps 300
    --max_time 00:20:00:00
    --load_ckpt /home/let55/workspace/projects/e3moldiffusion/geom/logs/qm9/run2/last.ckpt
    )

python train_full.py "${args[@]}"

