#!/bin/bash
#SBATCH -J TestEval
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --partition=ondemand-8xv100m32-1a
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch1/e3moldiffusion/slurm_logs_test/array_run_%j.out
#SBATCH --error=/scratch1/e3moldiffusion/slurm_logs_test/array_run_%j.err

cd /sharedhome/cremej01/workspace/e3moldiffusion
source activate e3mol
conda activate e3mol

export PYTHONPATH="/sharedhome/cremej01/workspace/e3moldiffusion"

main_dir="/scratch1/e3moldiffusion/logs/crossdocked/x0_snr_finetune_cutoff7_bonds5_out-norm_rbf-5A_edge-stuff_joint-sa"
output_dir="$main_dir/evaluation/test/nodes_bias_2"

mkdir "$main_dir/evaluation"
mkdir "$main_dir/evaluation/test"
mkdir "$output_dir"

python experiments/run_evaluation_ligand.py \
--model-path "$main_dir/best_valid.ckpt" \
--save-dir "$output_dir" \
--save-xyz \
--save-traj \
--batch-size 100 \
--dataset-root /scratch1/cremej01/data/crossdocked_noH_cutoff5_new_names \
--prior-n-atoms targetdiff \
--n-nodes-bias 2
#--use-ligand-dataset-sizes 
#--build-obabel-mol 

