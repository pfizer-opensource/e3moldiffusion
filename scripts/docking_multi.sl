#!/bin/bash
#SBATCH -J DockArray
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --partition=ondemand-cpu-c48
#SBATCH --array=1-20
#SBATCH --output=/scratch1/e3moldiffusion/slurm_logs/dock_array_run_%j.out
#SBATCH --error=/scratch1/e3moldiffusion/slurm_logs/dock_array_run_%j.err

cd /sharedhome/cremej01/workspace/e3moldiffusion
source activate e3mol
conda activate e3mol

export PYTHONPATH="/sharedhome/cremej01/workspace/e3moldiffusion"

main_dir="/scratch1/e3moldiffusion/logs/crossdocked/x0_snr_continous_sascore"
output_dir="$main_dir/evaluation/docking/nodes_bias_large_nosa"
mkdir "$output_dir/docked"

num_cpus=20

python experiments/docking_multi.py \
    --mp-index "${SLURM_ARRAY_TASK_ID}" \
    --num-cpus "$num_cpus" \
    --sdf-dir "$output_dir/sampled" \
    --save-dir "$output_dir" \
    --pdbqt-dir /scratch1/e3moldiffusion/data/crossdocked/crossdocked_5A_new/test/pdbqt \
    --pdb-dir /scratch1/e3moldiffusion/data/crossdocked/crossdocked_5A_new/test \
    --dataset crossdocked \
    --write-csv \
    --write-dict

# wait

# python experiments/aggregate_results.py \
#     --files-dir "$output_dir" \
#     --docked