#!/bin/bash
#SBATCH -J latent
#SBATCH --mail-user=tuan.le@pfizer.com
#SBATCH --mail-type=ALL
##SBATCH --partition=ondemand-8xv100m32-1a
#SBATCH --partition=gpu
#SBATCH --time=5-23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=25G
#SBATCH --cpus-per-task=16
##SBATCH --cpus-per-task=32
##SBATCH --gres=gpu:8
#SBATCH --nodelist=g003
#SBATCH --gres=gpu:4
##SBATCH --output=/sharedhome/let55/projects/e3moldiffusion_experiments/slurm_outs/latent_%j.out
##SBATCH --error=/sharedhome/let55/projects/e3moldiffusion_experiments/slurm_outs/latent_%j.err
#SBATCH --output=/hpfs/userws/let55/projects/e3moldiffusion_experiments/slurm_outs/latent_jointly_new_%j.out
#SBATCH --error=/hpfs/userws/let55/projects/e3moldiffusion_experiments/slurm_outs/latent_jointly_new_%j.err

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
