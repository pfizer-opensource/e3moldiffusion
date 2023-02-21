#!/bin/bash
#SBATCH -J drugs_experiment
#SBATCH --mail-user=tuan.le@pfizer.com 
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --constraint=weka
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 
#SBATCH --output=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/drugs_experiment_%j.out
#SBATCH --error=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/drugs_experiment_%j.err

cd /gpfs/workspace/users/let55/projects/e3moldiffusion/geom
source activate e3moldiffusion
echo "runnning experiment"


args=(
    --gpus 1 --id 0
    --dataset drugs
    --max_num_conformers 30 --num_workers 4
    --save_dir logs/drugs --num_epochs 100
    --sdim 128 --vdim 32 --tdim 128 --num_layers 4 
    --lr 5e-4 --batch_size 256
    --fully_connected 
    --use_bond_features --edim 32 
    --use_all_atom_features
    # --omit_norm
    --omit_cross_product
    --vector_aggr mean
    )

python train.py "${args[@]}"
