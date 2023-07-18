#!/bin/bash
#SBATCH -J drugs
#SBATCH --mail-user=tuan.le@pfizer.com
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --time=6-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=27G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --nodelist=g002
#SBATCH --output=/hpfs/userws/let55/projects/e3moldiffusion/experiments/geom/slurm_outs/drugs_experiment_%j.out
#SBATCH --error=/hpfs/userws/let55/projects/e3moldiffusion/experiments/geom/slurm_outs/drugs_experiment_%j.err

# delta
# cd /gpfs/workspace/users/let55/projects/e3moldiffusion
# alpha
cd /hpfs/userws/let55/projects/e3moldiffusion
# aws
# cd /sharedhome/let55/projects/e3moldiffusion/experiments/geom
source activate e3moldiffusion_new # alpha, delta
# source activate e3moldiffusion # aws
# python run_train.py --conf /hpfs/userws/let55/projects/e3moldiffusion/configs/diffusion_drugs_midi.yaml
python experiments/geom/run_train.py --conf /hpfs/userws/let55/projects/e3moldiffusion/configs/diffusion_drugs_DAE.yaml
# python run_train.py --conf /sharedhome/let55/projects/e3moldiffusion/configs/diffusion_drugs_edge_mp_midi.yaml
