#!/bin/bash
#SBATCH -J qm9_experiment
#SBATCH --mail-user=tuan.le@pfizer.com 
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --constraint=weka
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 
#SBATCH --output=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/qm9_experiment_%j.out
#SBATCH --error=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/qm9_experiment_%j.err

cd /gpfs/workspace/users/let55/projects/e3moldiffusion/geom
source activate e3moldiffusion
echo "runnning experiment"


args=(
    --gpus 1 --id 10 
    --dataset qm9
    --max_num_conformers 30 --num_workers 4
    --save_dir logs/qm9 --num_epochs 100
    --sdim 64 --vdim 16 --tdim 64 --num_layers 4 
    --lr 1e-4 --batch_size 256
    --fully_connected 
    --use_bond_features --edim 16 
    --use_all_atom_features
    )

python train.py "${args[@]}"
