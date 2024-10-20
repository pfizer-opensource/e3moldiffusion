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
pdb_file="2dq7"

main_dir=./model_folder
output_dir="$main_dir/evaluation/$pdb_file/your_out_folder"

mkdir "$main_dir/evaluation"
mkdir "$main_dir/evaluation/$pdb_file"
mkdir "$output_dir"

python experiments/generate_ligands_multi_batch.py \
    --mp-index "${SLURM_ARRAY_TASK_ID}" \
    --num-gpus "$num_gpus" \
    --model-path "$main_dir/best_valid.ckpt" \
    --save-dir "$output_dir" \
    --pdbqt-dir ./your_test_pdbqt_folder \
    --test-dir ./your_test_pdb_folder \
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
