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

num_gpus=15

cd /sharedhome/cremej01/workspace/e3moldiffusion
source activate e3mol
conda activate e3mol

export PYTHONPATH="/sharedhome/cremej01/workspace/e3moldiffusion"

main_dir="/scratch1/e3moldiffusion/logs/kiba/x0_snr_finetune_cutoff5_bonds5_norm_rbf-5A_edge-stuff_joint-sa-dock"
output_dir="$main_dir/evaluation/docking/nodes_bias_0_sa0-250_dock-250-350"

mkdir "$main_dir/evaluation"
mkdir "$main_dir/evaluation/docking"
mkdir "$output_dir"

python experiments/generate_ligands_multi.py \
    --mp-index "${SLURM_ARRAY_TASK_ID}" \
    --num-gpus "$num_gpus" \
    --model-path "$main_dir/best_valid.ckpt" \
    --save-dir "$output_dir" \
    --pdbqt-dir /scratch1/cremej01/data/kiba_noH_cutoff5/test/pdbqt \
    --test-dir /scratch1/cremej01/data/kiba_noH_cutoff5/test \
    --skip-existing \
    --num-ligands-per-pocket-to-sample 100 \
    --num-ligands-per-pocket-to-save 100 \
    --max-sample-iter 50 \
    --batch-size 40 \
    --n-nodes-bias 0 \
    --prior-n-atoms conditional \
    --property-importance-sampling \
    --property-importance-sampling-start 250 \
    --property-importance-sampling-end 350 \
    --property-every-importance-t 5 \
    --property-tau 0.1 \
    --sa-importance-sampling \
    --sa-importance-sampling-start 0 \
    --sa-importance-sampling-end 250 \
    --sa-every-importance-t 5 \
    --sa-tau 0.1 
    #--omit-posecheck \
    #--ckpts-ensemble /scratch1/e3moldiffusion/logs/kinodata/x0_snr_bonds5_cutoff5_norm_joint-sa-ic50_seed42/best_valid.ckpt /scratch1/e3moldiffusion/logs/kinodata/x0_snr_bonds5_cutoff5_norm_joint-sa-ic50_seed1000/best_valid.ckpt /scratch1/e3moldiffusion/logs/kinodata/x0_snr_bonds5_cutoff5_norm_joint-sa-ic50_seed500/best_valid.ckpt /scratch1/e3moldiffusion/logs/kinodata/x0_snr_bonds5_cutoff5_norm_joint-sa-ic50_seed800/best_valid.ckpt 
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