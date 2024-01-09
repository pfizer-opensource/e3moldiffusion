#!/bin/bash
#SBATCH -J SampleArray
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --partition=ondemand-8xv100m32-1b
#SBATCH --gres=gpu:1
#SBATCH --array=1-8
#SBATCH --output=/scratch1/e3moldiffusion/slurm_logs_multi/array_run_%j.out
#SBATCH --error=/scratch1/e3moldiffusion/slurm_logs_multi/array_run_%j.err

cd /sharedhome/cremej01/workspace/e3moldiffusion
source activate e3mol

export PYTHONPATH="/sharedhome/cremej01/workspace/e3moldiffusion"

main_dir="/scratch1/e3moldiffusion/logs/crossdocked/x0_snr_enamineft_cutoff5_bonds5_ep10"
output_dir="$main_dir/evaluation/docking/nodes_bias_large_vary"

mkdir "$main_dir/evaluation"
mkdir "$main_dir/evaluation/docking"
mkdir "$output_dir"

num_gpus=8

python experiments/generate_ligands_multi.py \
    --mp-index "${SLURM_ARRAY_TASK_ID}" \
    --num-gpus "$num_gpus" \
    --model-path "$main_dir/best_valid.ckpt" \
    --save-dir "$output_dir" \
    --test-dir /scratch1/cremej01/data/crossdocked_noH_cutoff5/test \
    --dataset-root /scratch1/e3moldiffusion/data/crossdocked/crossdocked_5A_new \
    --skip-existing \
    --num-ligands-per-pocket 100 \
    --batch-size 50 \
    --vary-n-nodes \
    --n-nodes-bias 10
    #--fix-n-nodes
    #--vary-n-nodes \
    #--build-obabel-mol \
    #--sanitize 
    #--relax-mol \
    #--max-relax-iter 500 

# for ((i=1; i<="$num_gpus"; i++)); do
#     file="${i}_posebusters_sampled.pickle"
    
#     while [ ! -f "$file" ]; do
#         sleep 5  # Adjust the interval between checks as needed
#     done
# done

# python experiments/aggregate_results.py \
#     --files-dir "$output_dir"