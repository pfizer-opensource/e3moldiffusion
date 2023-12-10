#!/bin/bash
#SBATCH -J ArrayJob
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=25G
#SBATCH --partition=ondemand-cpu-c48
#SBATCH --cpus-per-task=18
#SBATCH --array=1-50
#SBATCH --output=/scratch1/e3moldiffusion/logs/geom/slurm_outs/new_run_%j.out
#SBATCH --error=/scratch1/e3moldiffusion/logs/geom/slurm_outs/new_run_%j.err

cd /sharedhome/cremej01/workspace/e3moldiffusion
source activate /sharedhome/cremej01/workspace/mambaforge/envs/e3moldiffusion
export PYTHONPATH="${PYTHONPATH}:/sharedhome/cremej01/workspace/e3moldiffusion"

echo "runnning job array statistics calculation"
python experiments/data/calculate_energies.py --dataset drugs --split train --idx "${SLURM_ARRAY_TASK_ID}"
#srun python experiments/data/calculate_statistics.py --dataset pubchem --split train --idx "${SLURM_ARRAY_TASK_ID}"