#!/bin/bash
#SBATCH -J drugs_experiment
#SBATCH --mail-user=tuan.le@pfizer.com 
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --constraint=weka
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 
#SBATCH --output=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/experiment_%j.out
#SBATCH --error=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/experiment_%j.err

ls /hpfs/projects/mlcs/e3moldiffusion/drugs/database
cd /gpfs/workspace/users/let55/projects/e3moldiffusion/geom
source activate e3moldiffusion
echo "runnning experiment"

python train.py --gpus 1 --id 5 --max_num_conformers 30 --num_workers 4 --save_dir logs/drugs --num_epochs 100 --sdim 64 --vdim 16 --tdim 16 --num_layers 6 --lr 5e-4 --batch_size 256 --cutoff 7.0 --use_bond_features --edim 16 --energy_preserving