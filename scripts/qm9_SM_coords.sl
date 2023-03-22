#!/bin/bash
#SBATCH -J qm9_coords
#SBATCH --mail-user=tuan.le@pfizer.com
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu_medium
#SBATCH --constraint=weka
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 
#SBATCH --output=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/SM_qm9_%j.out
#SBATCH --error=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/SM_qm9_%j.err

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
    --save_dir logs/SM_qm9
    --num_epochs 100
    --sdim 64 --vdim 16 --edim 16 --rbf_dim 16 --num_layers 4 
    --cutoff 5.0
    --lr 5e-4
    --batch_size 256
    --local_global_model
    --dist_score
    --use_bond_features
    --use_all_atom_features
    --omit_cross_product
    --vector_aggr mean
    --max_time 00:07:50:00
    )

python train_score.py "${args[@]}"
