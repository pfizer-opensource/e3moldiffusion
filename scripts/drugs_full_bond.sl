#!/bin/bash
#SBATCH -J drugs_full
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
#SBATCH --output=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/01_drugs_fullatombond_%j.out
#SBATCH --error=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/01_drugs_fullatombond_%j.err

# attempting to access the data directory
ls /hpfs/projects/mlcs/e3moldiffusion

cd /gpfs/workspace/users/let55/projects/e3moldiffusion/geom
source activate e3moldiffusion
echo "runnning experiment"

args=(
    --gpus 1
    --id 6
    --dataset drugs
    --max_num_conformers 30
    --num_workers 4
    --save_dir logs/drugs_atomsbonds
    --num_epochs 100
    --sdim 128 --vdim 32 --rbf_dim 32 --num_layers 5 
    --cutoff_local 5.0
    --cutoff_global 10.0
    --lr 5e-4
    --batch_size 128
    --local_global_model
    --omit_cross_product
    --vector_aggr mean
    --schedule cosine
    --num_diffusion_timesteps 300
    --max_time 00:23:50:00
    )

python train_coordsatomsbonds.py "${args[@]}"