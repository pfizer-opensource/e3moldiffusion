#!/bin/bash
#SBATCH -J drugs_experiment
#SBATCH --mail-user=tuan.le@pfizer.com 
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1 
#SBATCH --output=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/experiment_%j.out
#SBATCH --error=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/experiment_%j.err

cd /gpfs/workspace/users/let55/projects/e3moldiffusion/geom
source activate e3moldiffusion
echo "runnning experiment"

python train.py --gpus 1 --id 1 --max_num_conformers 30 --num_workers 6 --save_dir logs/diffusion --num_epochs 100 \
 --load_ckpt /home/let55/workspace/projects/e3moldiffusion/geom/logs/diffusion/run0/last.ckpt