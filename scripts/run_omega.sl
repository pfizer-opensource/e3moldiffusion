#!/bin/bash
#SBATCH -J omega-conformer
#SBATCH --mail-user=tuan.le@pfizer.com
#SBATCH --mail-type=ALL
#SBATCH --partition=cpu
#SBATCH --time=10-23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=5
#SBATCH --output=/home/let55/workspace/datasets/enamine/out/slurm_out/result_%j.out
#SBATCH --error=/home/let55/workspace/datasets/enamine/out/slurm_out/result_%j.err
#SBATCH -a 0-241

cd /home/let55/workspace/datasets/enamine/out
module load medsci/2022.4
module load openeye/2022.1.2

oeomega classic -in chunk_${SLURM_ARRAY_TASK_ID}/smiles.smi -out chunk_${SLURM_ARRAY_TASK_ID}/out.sdf -maxconfs 5 -mpi_np 5