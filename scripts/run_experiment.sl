#!/bin/bash
#SBATCH -J newFeats
#SBATCH --mail-user=tuan.le@pfizer.com
#SBATCH --mail-type=ALL
#SBATCH --partition=ondemand-8xv100m32
##SBATCH --partition=gpu
#SBATCH --time=6-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=27G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:5
##SBATCH --nodelist=g002
#SBATCH --output=/sharedhome/let55/projects/e3moldiffusion/experiments/geom/slurm_outs/discrete_newFeats_12L_%j.out
#SBATCH --error=/sharedhome/let55/projects/e3moldiffusion/experiments/geom/slurm_outs/discrete_newFeats_12L_%j.err

# delta
# cd /gpfs/workspace/users/let55/projects/e3moldiffusion/experiments/geom
# alpha
# cd /hpfs/userws/let55/projects/e3moldiffusion/experiments/geom
# aws
cd /sharedhome/let55/projects/e3moldiffusion/experiments/geom
# source activate e3moldiffusion_new # alpha, delta
source activate e3moldiffusion # aws
python run_train.py --conf /sharedhome//let55/projects/e3moldiffusion/configs/diffusion_drugs.yaml
