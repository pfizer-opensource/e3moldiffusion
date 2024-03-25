#!/bin/bash
#SBATCH -J DockArray
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --partition=ondemand-cpu-c48
#SBATCH --array=1-40
#SBATCH --output=/scratch1/e3moldiffusion/slurm_logs/dock_array_run_%j.out
#SBATCH --error=/scratch1/e3moldiffusion/slurm_logs/dock_array_run_%j.err

num_cpus=40

cd /sharedhome/cremej01/workspace/e3moldiffusion
source activate e3mol
conda activate e3mol

export PYTHONPATH="/sharedhome/cremej01/workspace/e3moldiffusion"

main_dir="/scratch1/e3moldiffusion/logs/kinodata/x0_snr_bonds5_cutoff5_norm_joint-sa-ic50_seed1000"
output_dir="$main_dir/evaluation/docking/fix_nodes_bias_vary_5_sa0-350-every10_ic50-150-350"
mkdir "$output_dir/docked"


python experiments/docking_multi.py \
    --mp-index "${SLURM_ARRAY_TASK_ID}" \
    --num-cpus "$num_cpus" \
    --sdf-dir "$output_dir/sampled" \
    --save-dir "$output_dir" \
    --pdbqt-dir /scratch1/cremej01/data/kinodata_noH_cutoff5/test/pdbqt \
    --pdb-dir /scratch1/cremej01/data/kinodata_noH_cutoff5/test \
    --dataset kinodata \
    --write-csv \
    --write-dict

# wait

# python experiments/aggregate_results.py \
#     --files-dir "$output_dir" \
#     --docked