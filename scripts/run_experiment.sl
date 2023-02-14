#!/bin/bash
#SBATCH -J drugs_experiment
#SBATCH --mail-user=tuan.le@pfizer.com 
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu_medium
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4 
#SBATCH --output=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/experiment_%j.out
#SBATCH --error=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/experiment_%j.err

cd /gpfs/workspace/users/let55/projects/e3moldiffusion/geom
source activate e3moldiffusion
echo "runnning experiment"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python -c "import torch; print(torch.cuda.is_available())"

python train.py --gpus 4 --id 0 --subset_frac 1.0 --num_workers 4 --save_dir logs/diffusion --num_epochs 100