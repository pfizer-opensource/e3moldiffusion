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
source activate vina
conda activate vina

export PYTHONPATH="/sharedhome/cremej01/workspace/e3moldiffusion"

main_dir="/scratch1/e3moldiffusion/logs/kiba/x0_snr_finetune_cutoff5_bonds5_norm_rbf-5A_edge-stuff_joint-sa-dock"
output_dir="$main_dir/evaluation/docking/nodes_bias_0"

python experiments/docking_multi.py \
    --mp-index "${SLURM_ARRAY_TASK_ID}" \
    --num-cpus "$num_cpus" \
    --sdf-dir "$output_dir/sampled" \
    --save-dir "$output_dir" \
    --pdbqt-dir /scratch1/cremej01/data/kiba_noH_cutoff5/test/pdbqt \
    --pdb-dir /scratch1/cremej01/data/kiba_noH_cutoff5/test \
    --dataset kiba \
    --write-csv \
    --write-dict 
    #--docking-mode vina_score
