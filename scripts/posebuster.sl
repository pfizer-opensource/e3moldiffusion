#!/bin/bash -l
#SBATCH -J posbusting
#SBATCH --time=01-00:00:00
#SBATCH --nodes=1
#SBATCH --partition=ondemand-cpu-c48
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch1/e3moldiffusion/slurm_logs/out_%j.out
#SBATCH --error=/scratch1/e3moldiffusion/slurm_logs/err_%j.err

conda activate /sharedhome/cremej01/workspace/mambaforge/envs/e3moldiffusion
echo "runnning experiment"

cd /scratch1/e3moldiffusion/logs/cdk2/docking/addfeats_finetune_vary_nodes_cutoff8/raw

bust 4erw_STU_ligand_gen.sdf -p /scratch1/cremej01/data/cdk2/4erw.pdb --outfmt csv --output results.csv