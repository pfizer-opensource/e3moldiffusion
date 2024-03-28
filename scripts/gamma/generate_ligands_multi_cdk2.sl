#!/bin/bash
#SBATCH -J SampleArray
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --array=1-15
#SBATCH --output=/hpfs/projects/mlcs/mlhub/e3moldiffusion/logs/slurm_outs/array_run_%j.out
#SBATCH --error=/hpfs/projects/mlcs/mlhub/e3moldiffusion/logs/slurm_outs/array_run_%j.err

num_gpus=15

cd /hpfs/userws/cremej01/projects/e3moldiffusion
source /hpfs/userws/cremej01/projects/mambaforge/etc/profile.d/mamba.sh
source /hpfs/userws/cremej01/projects/mambaforge/etc/profile.d/conda.sh
conda activate e3mol

export PYTHONPATH="/hpfs/userws/cremej01/projects/e3moldiffusion"

main_dir="/hpfs/projects/mlcs/mlhub/e3moldiffusion/logs/molecular_glue/x0_snr_bonds5_cutoff5_pos-res_lig-pocket-inter_norm"
output_dir="$main_dir/evaluation/docking/fix_nodes_bias_vary_10_sa0-250_every-5"

mkdir "$main_dir/evaluation"
mkdir "$main_dir/evaluation/docking"
mkdir "$output_dir"

python experiments/generate_ligands_batch.py \
    --mp-index "${SLURM_ARRAY_TASK_ID}" \
    --num-gpus "$num_gpus" \
    --model-path "$main_dir/best_valid.ckpt" \
    --save-dir "$output_dir" \
    --pdbqt-dir /hpfs/projects/mlcs/mlhub/e3moldiffusion/data/cdk2/pdbqt \
    --test-dir /hpfs/projects/mlcs/mlhub/e3moldiffusion/data/cdk2 \
    --skip-existing \
    --num-ligands-to-sample 10000 \
    --max-sample-iter 50 \
    --batch-size 40 \
    --n-nodes-bias 10 \
    --vary-n-nodes \
    --fix-n-nodes \
    --sa-importance-sampling \
    --sa-importance-sampling-start 0 \
    --sa-importance-sampling-end 250 \
    --sa-every-importance-t 5 \
    --ckpt-sa-model /hpfs/projects/mlcs/mlhub/e3moldiffusion/logs/crossdocked/x0_snr_cutoff5_bonds5_no-norm_joint-sa/best_valid.ckpt \
    --sa-tau 0.1
    # --property-importance-sampling \
    # --property-importance-sampling-start 200 \
    # --property-importance-sampling-end 400 \
    # --property-every-importance-t 5 \
    # --property-tau 0.1 \
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
    #--docking-scores /hpfs/projects/mlcs/mlhub/e3moldiffusion/logs/crossdocked/ground_truth/evaluation/docking/crossdocked_scores.pickle
    #--filter-by-posebusters \
    #--filter-by-lipinski \
    #--filter-by-docking-scores \
    #--build-obabel-mol \
    #--sanitize 
    #--relax-mol \
    #--max-relax-iter 500 
