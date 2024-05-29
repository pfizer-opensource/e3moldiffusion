#!/bin/bash
#SBATCH -J DockArray
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --partition=ondemand-cpu-c48
#SBATCH --array=1-16
#SBATCH --output=/scratch1/e3moldiffusion/slurm_logs_dock/dock_array_run_%j.out
#SBATCH --error=/scratch1/e3moldiffusion/slurm_logs_dock/dock_array_run_%j.err

cd /sharedhome/cremej01/workspace/e3moldiffusion
source activate vina
conda activate vina
export PYTHONPATH="/sharedhome/cremej01/workspace/e3moldiffusion"

num_cpus=40
pdb_file="cdk2"

main_dir="/scratch1/e3moldiffusion/logs/crossdocked/x0_snr_cutoff5_bonds5_out-norm_rbf-5A_edge-stuff_joint-sa"
output_dir="$main_dir/evaluation/docking/$pdb_file/nodes_bias_vary_10_sa0-200_every-5"

python experiments/docking_multi.py \
    --mp-index "${SLURM_ARRAY_TASK_ID}" \
    --num-cpus "$num_cpus" \
    --sdf-dir "$output_dir/sampled" \
    --save-dir "$output_dir" \
    --pdbqt-dir "/scratch1/cremej01/data/$pdb_file" \
    --pdb-dir "/scratch1/cremej01/data/$pdb_file" \
    --dataset pdb_file \
    --write-csv \
    --write-dict \
    --docking-mode vina_dock
