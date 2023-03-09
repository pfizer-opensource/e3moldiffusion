#!/bin/bash
#SBATCH -J dataset_distibutions
#SBATCH --mail-user=tuan.le@pfizer.com
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu_short
#SBATCH --constraint=weka
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1 
#SBATCH --output=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/dataset_stats_%j.out
#SBATCH --error=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/dataset_stats_%j.err

# attempting to access the data directory
ls /hpfs/projects/mlcs/e3moldiffusion
cd /gpfs/workspace/users/let55/projects/e3moldiffusion/geom
source activate e3moldiffusion

args=(
    --dataset drugs
    --batch_size 256
    --num_workers 12
    --max_num_conformers 30
    --pin_memory True
    --persistent_workers True
)

python get_distributions.py "${args[@]}"
