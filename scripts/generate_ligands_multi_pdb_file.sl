#!/bin/bash
#SBATCH -J SampleArray
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --partition=ondemand-8xa100-1a
#SBATCH --gres=gpu:1
#SBATCH --array=1-15
#SBATCH --output=/scratch1/e3moldiffusion/slurm_logs_pdb/array_run_%j.out
#SBATCH --error=/scratch1/e3moldiffusion/slurm_logs_pdb/array_run_%j.err

cd /sharedhome/cremej01/workspace/e3moldiffusion
source activate e3mol
conda activate e3mol
export PYTHONPATH="/sharedhome/cremej01/workspace/e3moldiffusion"

num_gpus=15
pdb_file="2dq7"

main_dir="/scratch1/e3moldiffusion/logs/crossdocked/x0_snr_cutoff5_bonds5_out-norm_rbf-5A_edge-stuff_joint-sa"
output_dir="$main_dir/evaluation/docking/$pdb_file/nodes_bias_vary_10_sa0-200_every-5"

mkdir "$main_dir/evaluation"
mkdir "$main_dir/evaluation/docking"
mkdir "$main_dir/evaluation/docking/$pdb_file"
mkdir "$output_dir"

python experiments/generate_ligands_multi_batch.py \
    --mp-index "${SLURM_ARRAY_TASK_ID}" \
    --num-gpus "$num_gpus" \
    --model-path "$main_dir/best_valid.ckpt" \
    --save-dir "$output_dir" \
    --pdbqt-dir "/scratch1/cremej01/data/$pdb_file" \
    --test-dir "/scratch1/cremej01/data/$pdb_file" \
    --skip-existing \
    --num-ligands-per-pocket-to-sample 10000 \
    --num-ligands-per-pocket-to-save 10000 \
    --max-sample-iter 50 \
    --batch-size 40 \
    --prior-n-atoms conditional \
    --n-nodes-bias 10 \
    --vary-n-nodes \
    --sa-importance-sampling \
    --sa-importance-sampling-start 0 \
    --sa-importance-sampling-end 200 \
    --sa-every-importance-t 5 \
    --sa-tau 0.1 \
    --dataset-root /scratch1/cremej01/data/crossdocked_noH_cutoff5_new
    # --property-importance-sampling \
    # --property-importance-sampling-start 0 \
    # --property-importance-sampling-end 200 \
    # --property-every-importance-t 10 \
    # --property-tau 0.5 \
    # --property-normalization \
    #--dataset-root /scratch1/cremej01/data/crossdocked_noH_cutoff5_qvina2-dock
    # --ckpt-property-model None \
    # --minimize-property
    # --property-classifier-guidance None \
    # --property-classifier-guidance_complex False \
    # --property_classifier_self_guidance False \
    # --classifier_guidance_scale None \
    # --fix-n-nodes \
    # --encode-ligands \
    # --filter-by-sascore \
    # --sascore-threshold 0.6
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