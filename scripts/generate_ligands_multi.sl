#!/bin/bash
#SBATCH -J SampleArray
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --partition=your_partition
#SBATCH --gres=gpu:1
#SBATCH --array=1-15
#SBATCH --output=./your_log_folder/array_run_%j.out
#SBATCH --error=./your_log_folder/array_run_%j.err

cd ./pilot_folder
source activate your_env
conda activate your_env
export PYTHONPATH=./pilot_folder

num_gpus=15

main_dir=./model_folder
output_dir="$main_dir/evaluation/your_out_folder"

mkdir "$main_dir/evaluation"
mkdir "$output_dir"

python experiments/generate_ligands_multi.py \
    --mp-index "${SLURM_ARRAY_TASK_ID}" \
    --num-gpus "$num_gpus" \
    --model-path "$main_dir/best_valid.ckpt" \
    --save-dir "$output_dir" \
    --pdbqt-dir ./your_test_pdbqt_folder \
    --test-dir ./your_test_pdb_folder \
    --dataset-root ./your_dataset_folder \
    --skip-existing \
    --num-ligands-per-pocket-to-sample 100 \
    --num-ligands-per-pocket-to-save 100 \
    --max-sample-iter 50 \
    --batch-size 40 \
    --n-nodes-bias 2 \
    --latent-gamma 1.0 \
    --encode-ligands \
    # --fix-n-nodes
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

