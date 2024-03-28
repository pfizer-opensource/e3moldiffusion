#!/bin/bash
#SBATCH -J DockArray
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --partition=defq
#SBATCH --array=1-40
#SBATCH --output=/hpfs/projects/mlcs/mlhub/e3moldiffusion/logs/slurm_outs_dock/dock_array_run_%j.out
#SBATCH --error=/hpfs/projects/mlcs/mlhub/e3moldiffusion/logs/slurm_outs_dock/dock_array_run_%j.err

num_cpus=40

cd /hpfs/userws/cremej01/projects/e3moldiffusion
source /hpfs/userws/cremej01/projects/mambaforge/etc/profile.d/mamba.sh
source /hpfs/userws/cremej01/projects/mambaforge/etc/profile.d/conda.sh
conda activate e3mol

export PYTHONPATH="/hpfs/userws/cremej01/projects/e3moldiffusion"

main_dir="/hpfs/projects/mlcs/mlhub/e3moldiffusion/data/kinodata_noH_cutoff5/val"
output_dir="$main_dir/evaluation/docking"
mkdir "$output_dir/docked"

python experiments/docking_multi.py \
    --mp-index "${SLURM_ARRAY_TASK_ID}" \
    --num-cpus "$num_cpus" \
    --sdf-dir "$main_dir" \
    --save-dir "$output_dir" \
    --pdbqt-dir /hpfs/projects/mlcs/mlhub/e3moldiffusion/data/kinodata_noH_cutoff5/val/pdbqt \
    --pdb-dir /hpfs/projects/mlcs/mlhub/e3moldiffusion/data/kinodata_noH_cutoff5/val \
    --dataset kinodata \
    --write-csv \
    --write-dict \
    --avoid-eval
