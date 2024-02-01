#!/bin/bash
#SBATCH -J SampleArray
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --partition=ondemand-8xv100m32-1a
#SBATCH --gres=gpu:1
#SBATCH --array=1-15
#SBATCH --output=/scratch1/e3moldiffusion/slurm_logs_multi/array_run_%j.out
#SBATCH --error=/scratch1/e3moldiffusion/slurm_logs_multi/array_run_%j.err

cd /sharedhome/cremej01/workspace/e3moldiffusion
source activate e3mol
conda activate e3mol

export PYTHONPATH="/sharedhome/cremej01/workspace/e3moldiffusion"

main_dir="/scratch1/e3moldiffusion/logs/crossdocked/x0_snr_continous_sascore"
output_dir="$main_dir/evaluation/docking/nodes_bias_large_sa"

mkdir "$main_dir/evaluation"
mkdir "$main_dir/evaluation/docking"
mkdir "$output_dir"

num_gpus=15

python experiments/generate_ligands_multi.py \
    --mp-index "${SLURM_ARRAY_TASK_ID}" \
    --num-gpus "$num_gpus" \
    --model-path "$main_dir/best_valid.ckpt" \
    --save-dir "$output_dir" \
    --pdbqt-dir /scratch1/e3moldiffusion/data/crossdocked/crossdocked_5A_new/test/pdbqt \
    --test-dir /scratch1/e3moldiffusion/data/crossdocked/crossdocked_5A_new/test \
    --dataset-root /scratch1/e3moldiffusion/data/crossdocked/crossdocked_5A_new \
    --skip-existing \
    --num-ligands-per-pocket 100 \
    --max-sample-iter 40 \
    --batch-size 8 \
    --n-nodes-bias 10 \
    --property-self-guidance \
    --guidance-scale 1.
    #--fix-n-nodes \
    #--vary-n-nodes \
    # --encode-ligand \
    #--omit-posebusters \
    #--omit-posecheck \
    #--docking-scores /scratch1/e3moldiffusion/logs/crossdocked/ground_truth/evaluation/docking/crossdocked_scores.pickle
    #--filter-by-posebusters \
    # --filter-by-lipinski \
    # --filter-by-docking-scores \
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