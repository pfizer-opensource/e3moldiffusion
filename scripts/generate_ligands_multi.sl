#!/bin/bash
#SBATCH -J SampleArray
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --partition=ondemand-8xv100m32-1b
#SBATCH --gres=gpu:1
#SBATCH --array=1-16
#SBATCH --output=/scratch1/e3moldiffusion/slurm_logs_multi/array_run_%j.out
#SBATCH --error=/scratch1/e3moldiffusion/slurm_logs_multi/array_run_%j.err

num_gpus=16

cd /sharedhome/cremej01/workspace/e3moldiffusion
source activate e3mol
conda activate e3mol

export PYTHONPATH="/sharedhome/cremej01/workspace/e3moldiffusion"

main_dir="/scratch1/e3moldiffusion/logs/crossdocked/x0_snr_bonds5_cutoff5_pos-res_lig-pocket-inter_norm_joint-sa-dock"
output_dir="$main_dir/evaluation/docking/nodes_bias_vary_10_sa0-450_dock-200-400"

mkdir "$main_dir/evaluation"
mkdir "$main_dir/evaluation/docking"
mkdir "$output_dir"

python experiments/generate_ligands_multi.py \
    --mp-index "${SLURM_ARRAY_TASK_ID}" \
    --num-gpus "$num_gpus" \
    --model-path "$main_dir/best_valid.ckpt" \
    --save-dir "$output_dir" \
    --pdbqt-dir /scratch1/cremej01/data/crossdocked_noH_cutoff5_new/test/pdbqt \
    --test-dir /scratch1/cremej01/data/crossdocked_noH_cutoff5_new/test \
    --skip-existing \
    --num-ligands-per-pocket-to-sample 100 \
    --num-ligands-per-pocket-to-save 100 \
    --max-sample-iter 50 \
    --batch-size 40 \
    --n-nodes-bias 10 \
    --vary-n-nodes \
    --importance-sampling \
    --importance-sampling-start 0 \
    --importance-sampling-end 450 \
    --every-importance-t 5 \
    --tau 0.1 \
    --docking-guidance \
    --docking-t-start 200 \
    --docking-t-end 400 \
    --tau1 0.1
    # --fix-n-nodes \
    # --encode-ligands \
    # --filter-by-sascore \
    # --sascore-threshold 0.6
    # --property-guidance-complex \
    # --ckpt-property-model /scratch1/e3moldiffusion/logs/crossdocked/docking_score_training/run0/last-v5.ckpt \
    # --guidance-scale 1.
    #--property-guidance \
    #--ckpt-property-model /scratch1/e3moldiffusion/logs/crossdocked/sascore_training/run0/last-v11.ckpt \
    #--guidance-scale 1.
    #--omit-posebusters \
    #--omit-posecheck \
    #--docking-scores /scratch1/e3moldiffusion/logs/crossdocked/ground_truth/evaluation/docking/crossdocked_scores.pickle
    #--filter-by-posebusters \
    #--filter-by-lipinski \
    #--filter-by-docking-scores \
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