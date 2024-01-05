#!/bin/bash -l
#SBATCH -J aggregate
#SBATCH --time=00-01:00:00
#SBATCH --nodes=1
#SBATCH --partition=ondemand-cpu-c48
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --output=/scratch1/e3moldiffusion/slurm_logs/aggr_out_%j.out
#SBATCH --error=/scratch1/e3moldiffusion/slurm_logs/aggr_err_%j.err

cd /sharedhome/cremej01/workspace/e3moldiffusion
source activate e3mol

export PYTHONPATH="/sharedhome/cremej01/workspace/e3moldiffusion"

main_dir="/scratch1/e3moldiffusion/logs/crossdocked/x0_snr_enamineft_cutoff5_bonds5_ep10"
output_dir="$main_dir/evaluation/docking/nodes_bias_large_multi"

python experiments/aggregate_results.py \
    --files-dir "$output_dir"
    #--docked