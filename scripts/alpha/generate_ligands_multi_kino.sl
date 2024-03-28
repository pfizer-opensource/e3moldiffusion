#!/bin/bash
#SBATCH -J SampleArray
#SBATCH --time=00-23:59:59
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu_medium
#SBATCH --gres=gpu:1
#SBATCH --array=1-3
#SBATCH --output=/hpfs/userws/cremej01/projects/logs/slurm_outs/array_run_%j.out
#SBATCH --error=/hpfs/userws/cremej01/projects/logs/slurm_outs/array_run_%j.err

num_gpus=3

cd /hpfs/userws/cremej01/projects/e3moldiffusion
source /hpfs/userws/cremej01/mambaforge/etc/profile.d/mamba.sh
source /hpfs/userws/cremej01/mambaforge/etc/profile.d/conda.sh
conda activate e3mol

export PYTHONPATH="/hpfs/userws/cremej01/projects/e3moldiffusion"

main_dir="/hpfs/userws/cremej01/projects/logs/kinodata/x0_snr_bonds5_cutoff5_norm_joint-sa-ic50_seed42"
output_dir="$main_dir/evaluation/docking/fix_nodes_bias_vary_6_sa0-250-every10_ic50-150-400_ensemble"

mkdir "$main_dir/evaluation"
mkdir "$main_dir/evaluation/docking"
mkdir "$output_dir"

python experiments/generate_ligands_multi.py \
    --mp-index "${SLURM_ARRAY_TASK_ID}" \
    --num-gpus "$num_gpus" \
    --model-path "$main_dir/best_valid.ckpt" \
    --save-dir "$output_dir" \
    --pdbqt-dir /hpfs/userws/cremej01/projects/data/kinodata_noH_cutoff5/test/pdbqt \
    --test-dir /hpfs/userws/cremej01/projects/data/kinodata_noH_cutoff5/test \
    --dataset-root /hpfs/userws/cremej01/projects/data/kinodata_noH_cutoff5 \
    --skip-existing \
    --num-ligands-per-pocket-to-sample 100 \
    --num-ligands-per-pocket-to-save 100 \
    --max-sample-iter 50 \
    --batch-size 60 \
    --fix-n-nodes \
    --n-nodes-bias 6 \
    --vary-n-nodes \
    --prior-n-atoms conditional \
    --property-importance-sampling \
    --property-importance-sampling-start 150 \
    --property-importance-sampling-end 400 \
    --property-every-importance-t 5 \
    --property-tau 0.1 \
    --sa-importance-sampling \
    --sa-importance-sampling-start 0 \
    --sa-importance-sampling-end 250 \
    --sa-every-importance-t 10 \
    --sa-tau 0.1 \
    --omit-posecheck \
    --ckpts-ensemble /hpfs/userws/cremej01/projects/logs/kinodata/x0_snr_bonds5_cutoff5_norm_joint-sa-ic50_seed42/best_valid.ckpt /hpfs/userws/cremej01/projects/logs/kinodata/x0_snr_bonds5_cutoff5_norm_joint-sa-ic50_seed1000/best_valid.ckpt /hpfs/userws/cremej01/projects/logs/kinodata/x0_snr_bonds5_cutoff5_norm_joint-sa-ic50_seed500/best_valid.ckpt
    # --sa-importance-sampling \
    # --sa-importance-sampling-start 0 \
    # --sa-importance-sampling-end 200 \
    # --sa-every-importance-t 5 \
    # --sa-tau 0.1 \
    # --ckpt-sa-model None \
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