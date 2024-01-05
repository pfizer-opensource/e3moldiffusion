#!/bin/bash
#SBATCH -J ArrayJob
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --partition=ondemand-8xv100m32-1a
#SBATCH --gres=gpu:1
#SBATCH --array=1-8
#SBATCH --output=/scratch1/e3moldiffusion/slurm_logs/array_run_%j.out
#SBATCH --error=/scratch1/e3moldiffusion/slurm_logs/array_run_%j.err

cd /sharedhome/cremej01/workspace/e3moldiffusion
source activate e3mol

export PYTHONPATH="/sharedhome/cremej01/workspace/e3moldiffusion"

main_dir="/scratch1/e3moldiffusion/logs/crossdocked/x0_snr_finetune_cutoff5_bonds7_addfeats"
output_dir="$main_dir/evaluation/docking/nodes_bias_large_multi"

mkdir "$main_dir/evaluation"
mkdir "$main_dir/evaluation/docking"
mkdir "$output_dir"

python experiments/generate_ligands_multi.py \
    --mp-index "${SLURM_ARRAY_TASK_ID}" \
    --num-gpus 8 \
    --model-path "$main_dir/best_valid.ckpt" \
    --save-dir "$output_dir" \
    --test-dir /scratch1/cremej01/data/crossdocked_noH_cutoff5/test \
    --pdb-dir /scratch1/cremej01/data/crossdocked_pdbs \
    --dataset-root /scratch1/cremej01/data/crossdocked_noH_cutoff5 \
    --skip-existing \
    --num-ligands-per-pocket 100 \
    --batch-size 50 \
    --vary-n-nodes \
    --n-nodes-bias 10
    #--fix-n-nodes
    #--vary-n-nodes \
    #--build-obabel-mol \
    #--sanitize 
    #--relax-mol \
    #--max-relax-iter 500 

if [ "${SLURM_ARRAY_TASK_ID}" -eq 8 ]; then
    sbatch scripts/aggregate_results.sh
fi