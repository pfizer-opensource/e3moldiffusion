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
main_dir="/scratch1/e3moldiffusion/logs/crossdocked/x0_snr_cutoff5_bonds5_norm_joint-sa-dock_Old"
output_dir="$main_dir/evaluation/docking/nodes_bias_vary_10_dock200-300"

python experiments/docking_multi.py \
    --mp-index "${SLURM_ARRAY_TASK_ID}" \
    --num-cpus "$num_cpus" \
    --sdf-dir "$output_dir/sampled" \
    --save-dir "$output_dir" \
    --pdbqt-dir /scratch1/e3moldiffusion/data/crossdocked/crossdocked_noH_cutoff5_new/test/pdbqt \
    --pdb-dir /scratch1/e3moldiffusion/data/crossdocked/crossdocked_noH_cutoff5_new/test \
    --dataset crossdocked \
    --write-csv \
    --write-dict \
    --docking-mode vina_dock
