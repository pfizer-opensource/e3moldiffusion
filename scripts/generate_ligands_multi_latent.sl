#!/bin/bash
#SBATCH -J SampleArray
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --partition=ondemand-8xv100m32-1b
#SBATCH --gres=gpu:1
#SBATCH --array=1-16
#SBATCH --output=/scratch1/e3moldiffusion/slurm_logs_generate/array_run_%j.out
#SBATCH --error=/scratch1/e3moldiffusion/slurm_logs_generate/array_run_%j.err

num_gpus=16

cd /sharedhome/cremej01/workspace/e3moldiffusion
source activate e3mol
conda activate e3mol

export PYTHONPATH="/sharedhome/cremej01/workspace/e3moldiffusion"

main_dir="/scratch1/e3moldiffusion/logs/crossdocked/x0_snr_finetune_cutoff5_bonds5_norm_rbf-5A_edge-stuff_latent_joint-sa"
output_dir="$main_dir/evaluation/docking/fix_n_nodes_vary_6"

mkdir "$main_dir/evaluation"
mkdir "$main_dir/evaluation/docking"
mkdir "$output_dir"

python experiments/generate_ligands_multi.py \
    --mp-index "${SLURM_ARRAY_TASK_ID}" \
    --num-gpus "$num_gpus" \
    --model-path "$main_dir/best_valid.ckpt" \
    --save-dir "$output_dir" \
    --pdbqt-dir /scratch1/e3moldiffusion/data/crossdocked/crossdocked_noH_cutoff5_new/test/pdbqt \
    --test-dir /scratch1/e3moldiffusion/data/crossdocked/crossdocked_noH_cutoff5_new/test \
    --dataset-root /scratch1/e3moldiffusion/data/crossdocked/crossdocked_noH_cutoff5_new \
    --skip-existing \
    --num-ligands-per-pocket-to-sample 100 \
    --num-ligands-per-pocket-to-save 100 \
    --max-sample-iter 50 \
    --batch-size 40 \
    --n-nodes-bias 6 \
    --fix-n-nodes \
    --encode-ligands \
    --latent-gamma 1.0
    # --prior-n-atoms targetdiff \
    # --omit-posecheck
    # --vary-n-nodes
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
    # --ckpt-property-model /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_bonds5_cutoff5_pos-res_lig-pocket-inter_no-norm_joint-dock/best_valid.ckpt \
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
    #--docking-scores /scratch1/e3moldiffusion/logs/crossdocked/ground_truth/evaluation/docking/crossdocked_scores.pickle
    #--filter-by-posebusters \
    #--filter-by-lipinski \
    #--filter-by-docking-scores \
    #--build-obabel-mol \
    #--sanitize 
    #--relax-mol \
    #--max-relax-iter 500 
