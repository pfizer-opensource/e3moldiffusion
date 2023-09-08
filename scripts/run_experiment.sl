#!/bin/bash
#SBATCH -J pretrain_sbatch
#SBATCH --mail-user=tuan.le@pfizer.com
#SBATCH --mail-type=ALL
#SBATCH --partition=ondemand-8xa100
#SBATCH --time=5-23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=25G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --nodelist=ondemand-8xa100-dy-node-4
#SBATCH --output=/sharedhome/let55/projects/e3moldiffusion_experiments/slurm_outs/pretrain_bs50_%j.out
#SBATCH --error=/sharedhome/let55/projects/e3moldiffusion_experiments/slurm_outs/pretrain_bs50_%j.err

# delta
# cd /gpfs/workspace/users/let55/projects/e3moldiffusion/experiments/
# alpha
# cd /hpfs/userws/let55/projects/e3moldiffusion/experiments/
# aws
cd /sharedhome/let55/projects/e3moldiffusion/
source activate e3moldiffusion
#python experiments/run_train.py --conf /sharedhome/let55/projects/e3moldiffusion/configs/diffusion_drugs_latent.yaml
#python experiments/run_train.py --conf /hpfs/userws/let55/projects/e3moldiffusion/configs/diffusion_drugs_latent.yaml

# python experiments/run_train.py --conf /hpfs/userws/let55/projects/e3moldiffusion/configs/diffusion_drugs.yaml
# python experiments/run_train.py --conf /sharedhome/let55/projects/e3moldiffusion/configs/diffusion_drugs.yaml
# python experiments/un_train.py --conf /sharedhome/let55/projects/e3moldiffusion/configs/diffusion_qm9.yaml
# python experiments/run_train.py --conf /hpfs/userws/let55/projects/e3moldiffusion/configs/diffusion_qm9.yaml


# python experiments/run_train.py --conf configs/diffusion_qm9.yaml
# python experiments/run_train.py --conf configs/diffusion_drugs.yaml
python experiments/run_train.py --conf configs/diffusion_pubchem.yaml