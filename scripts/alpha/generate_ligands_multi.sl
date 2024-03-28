#!/bin/bash
#SBATCH -J SampleArray
#SBATCH --time=00-23:59:59
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu_medium
#SBATCH --gres=gpu:1
#SBATCH --array=1-15
#SBATCH --output=/hpfs/userws/cremej01/projects/logs/slurm_outs/array_run_%j.out
#SBATCH --error=/hpfs/userws/cremej01/projects/logs/slurm_outs/array_run_%j.err

num_gpus=15

cd /hpfs/userws/cremej01/projects/e3moldiffusion
source /hpfs/userws/cremej01/mambaforge/etc/profile.d/mamba.sh
source /hpfs/userws/cremej01/mambaforge/etc/profile.d/conda.sh
conda activate e3mol

export PYTHONPATH="/hpfs/userws/cremej01/projects/e3moldiffusion"

main_dir="/hpfs/userws/cremej01/projects/logs/crossdocked/x0_snr_cutoff5_bonds5_no-norm_rbf"
output_dir="$main_dir/evaluation/docking/nodes_bias_vary_10"

mkdir "$main_dir/evaluation"
mkdir "$main_dir/evaluation/docking"
mkdir "$output_dir"

python experiments/generate_ligands_multi.py \
    --mp-index "${SLURM_ARRAY_TASK_ID}" \
    --num-gpus "$num_gpus" \
    --model-path "$main_dir/best_valid.ckpt" \
    --save-dir "$output_dir" \
    --pdbqt-dir /hpfs/userws/cremej01/projects/data/crossdocked_noH_cutoff5_dock_new/test/pdbqt \
    --test-dir /hpfs/userws/cremej01/projects/data/crossdocked_noH_cutoff5_dock_new/test \
    --skip-existing \
    --num-ligands-per-pocket-to-sample 100 \
    --num-ligands-per-pocket-to-save 100 \
    --max-sample-iter 50 \
    --batch-size 40 \
    --n-nodes-bias 10 \
    --vary-n-nodes \
    --prior-n-atoms targetdiff \
    --omit-posecheck
    # --property-importance-sampling \
    # --property-importance-sampling-start 200 \
    # --property-importance-sampling-end 300 \
    # --property-every-importance-t 5 \
    # --property-tau 0.1 \
    # --sa-importance-sampling \
    # --sa-importance-sampling-start 0 \
    # --sa-importance-sampling-end 300 \
    # --sa-every-importance-t 5 \
    # --sa-tau 0.1 
    # --vary-n-nodes \
    # --ckpt-property-model /hpfs/userws/cremej01/projects/logs/crossdocked/x0_snr_bonds5_cutoff5_pos-res_lig-pocket-inter_no-norm_joint-dock/best_valid.ckpt \
    # --ckpt-sa-model None \
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
    #--docking-scores /hpfs/userws/cremej01/projects/logs/crossdocked/ground_truth/evaluation/docking/crossdocked_scores.pickle
    #--filter-by-posebusters \
    #--filter-by-lipinski \
    #--filter-by-docking-scores \
    #--build-obabel-mol \
    #--sanitize 
    #--relax-mol \
    #--max-relax-iter 500 
