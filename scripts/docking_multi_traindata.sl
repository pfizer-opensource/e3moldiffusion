#!/bin/bash
#SBATCH -J DockArray
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --partition=ondemand-cpu-c48
#SBATCH --array=1-50
#SBATCH --output=/scratch1/e3moldiffusion/slurm_logs/dock_array_run_%j.out
#SBATCH --error=/scratch1/e3moldiffusion/slurm_logs/dock_array_run_%j.err

num_cpus=50

cd /sharedhome/cremej01/workspace/e3moldiffusion
source activate e3mol
conda activate e3mol

export PYTHONPATH="/sharedhome/cremej01/workspace/e3moldiffusion"

main_dir="/scratch1/cremej01/data/kinodata_noH_cutoff5/train"
output_dir="$main_dir/evaluation/docking"
mkdir "$output_dir/docked"


python experiments/docking_multi.py \
    --mp-index "${SLURM_ARRAY_TASK_ID}" \
    --num-cpus "$num_cpus" \
    --sdf-dir "$main_dir" \
    --save-dir "$output_dir" \
    --pdbqt-dir /scratch1/cremej01/data/kinodata_noH_cutoff5/train/pdbqt \
    --pdb-dir /scratch1/cremej01/data/kinodata_noH_cutoff5/train \
    --dataset kinodata \
    --write-csv \
    --write-dict \
    --avoid-eval
