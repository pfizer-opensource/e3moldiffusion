#!/bin/bash
#SBATCH -J SampleArray
#SBATCH --time=01-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --array=1-4
#SBATCH --output=/hpfs/projects/mlcs/mlhub/e3moldiffusion/logs/slurm_outs/array_run_%j.out
#SBATCH --error=/hpfs/projects/mlcs/mlhub/e3moldiffusion/logs/slurm_outs/array_run_%j.err

num_gpus=4

cd /hpfs/userws/cremej01/projects/e3moldiffusion
source /hpfs/userws/cremej01/projects/mambaforge/etc/profile.d/mamba.sh
source /hpfs/userws/cremej01/projects/mambaforge/etc/profile.d/conda.sh
conda activate e3mol

export PYTHONPATH="/hpfs/userws/cremej01/projects/e3moldiffusion"


main_dir="/hpfs/projects/mlcs/mlhub/e3moldiffusion/logs/crossdocked/x0_snr_cutoff5_bonds5_norm_rbf_edge-rbf_global-edge_cutoff-damping"
output_dir="$main_dir/evaluation/docking/nodes_bias_5"

mkdir "$main_dir/evaluation"
mkdir "$main_dir/evaluation/docking"
mkdir "$output_dir"

python experiments/generate_ligands_multi.py \
    --mp-index "${SLURM_ARRAY_TASK_ID}" \
    --num-gpus "$num_gpus" \
    --model-path "$main_dir/best_valid.ckpt" \
    --save-dir "$output_dir" \
    --pdbqt-dir /hpfs/projects/mlcs/mlhub/e3moldiffusion/data/crossdocked/crossdocked_noH_cutoff6_TargetDiff_atmass/test/pdbqt \
    --test-dir /hpfs/projects/mlcs/mlhub/e3moldiffusion/data/crossdocked/crossdocked_noH_cutoff6_TargetDiff_atmass/test \
    --skip-existing \
    --num-ligands-per-pocket-to-sample 100 \
    --num-ligands-per-pocket-to-save 100 \
    --max-sample-iter 50 \
    --batch-size 80 \
    --n-nodes-bias 5 \
    --prior-n-atoms targetdiff \
    --omit-posecheck
    # --vary-n-nodes \
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
    # --ckpt-property-model /hpfs/projects/mlcs/mlhub/e3moldiffusion/logs/crossdocked/x0_snr_bonds5_cutoff5_pos-res_lig-pocket-inter_no-norm_joint-dock/best_valid.ckpt \
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
    #--docking-scores /hpfs/projects/mlcs/mlhub/e3moldiffusion/logs/crossdocked/ground_truth/evaluation/docking/crossdocked_scores.pickle
    #--filter-by-posebusters \
    #--filter-by-lipinski \
    #--filter-by-docking-scores \
    #--build-obabel-mol \
    #--sanitize 
    #--relax-mol \
    #--max-relax-iter 500 
