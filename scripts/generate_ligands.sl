#!/bin/bash -l
#SBATCH -J addfeats_c5_b7
#SBATCH --time=02-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch1/e3moldiffusion/slurm_logs/out_%j.out
#SBATCH --error=/scratch1/e3moldiffusion/slurm_logs/err_%j.err

echo "Job ID: $SLURM_JOB_ID"

cd /sharedhome/cremej01/workspace/e3moldiffusion
conda activate /sharedhome/cremej01/workspace/mambaforge/envs/e3moldiffusion

export PYTHONPATH="/sharedhome/cremej01/workspace/e3moldiffusion"

main_dir="/scratch1/e3moldiffusion/logs/crossdocked/x0_snr_addfeats_cutoff5_bonds7"
output_dir="$main_dir/evaluation/docking/nodes_bias_large"
result_file="$output_dir/finished"

mkdir "$main_dir/evaluation"
mkdir "$main_dir/evaluation/docking"
mkdir "$output_dir"

python experiments/generate_ligands.py \
    --model-path "$main_dir/best_valid.ckpt" \
    --save-dir "$output_dir" \
    --test-dir /scratch1/cremej01/data/crossdocked_noH_cutoff5/test \
    --dataset-root /scratch1/cremej01/data/crossdocked_noH_cutoff5 \
    --skip-existing \
    --num-ligands-per-pocket 100 \
    --batch-size 50 \
    --n-nodes-bias 10
    #--fix-n-nodes
    #--vary-n-nodes \
    #--build-obabel-mol \
    #--sanitize 
    #--relax-mol \
    #--max-relax-iter 500 

touch result_file
