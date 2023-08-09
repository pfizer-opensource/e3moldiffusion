#!/bin/bash
#SBATCH -J LatentDiffusion
#SBATCH --mail-user=tuan.le@pfizer.com
#SBATCH --mail-type=ALL
##SBATCH --partition=ondemand-8xv100m32-1a
#SBATCH --partition=gpu
#SBATCH --time=1-23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=25G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --nodelist=g003
##SBATCH --output=/sharedhome/let55/projects/e3moldiffusion_experiments/slurm_outs/Discrete_12L_noH_%j.out
##SBATCH --error=/sharedhome/let55/projects/e3moldiffusion_experiments/slurm_outs/Discrete_12L_noH_%j.err
#SBATCH --output=/hpfs/userws/let55/projects/e3moldiffusion_experiments/slurm_outs/LatentDIFFUSION_%j.out
#SBATCH --error=/hpfs/userws/let55/projects/e3moldiffusion_experiments/slurm_outs/LatentDIFFUSION_%j.err

# delta
# cd /gpfs/workspace/users/let55/projects/e3moldiffusion/experiments/
# alpha
cd /hpfs/userws/let55/projects/e3moldiffusion/experiments/
# aws
# cd /sharedhome/let55/projects/e3moldiffusion/experiments/
source activate e3moldiffusion
# python run_train.py --conf /sharedhome/let55/projects/e3moldiffusion/configs/diffusion_drugs_latent.yaml
python run_train.py --conf /hpfs/userws/let55/projects/e3moldiffusion/configs/diffusion_drugs_latent.yaml

# python run_train.py --conf /hpfs/userws/let55/projects/e3moldiffusion/configs/diffusion_drugs.yaml
# python run_train.py --conf /sharedhome/let55/projects/e3moldiffusion/configs/diffusion_drugs.yaml
