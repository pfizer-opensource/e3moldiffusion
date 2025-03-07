#!/bin/bash
#SBATCH -J DockingArray
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --partition=your_partition
#SBATCH --array=1-40
#SBATCH --output=./your_log_folder/array_run_%j.out
#SBATCH --error=./your_log_folder/array_run_%j.err

cd ./pilot_folder
source activate vina
conda activate vina
export PYTHONPATH=./pilot_folder

num_cpus=40
pdb_file="cdk2"


main_dir=./model_folder
output_dir="$main_dir/evaluation/$pdb_file/your_out_folder"

mkdir "$main_dir/evaluation"
mkdir "$main_dir/evaluation/$pdb_file"
mkdir "$output_dir"


python experiments/docking_multi.py \
    --mp-index "${SLURM_ARRAY_TASK_ID}" \
    --num-cpus "$num_cpus" \
    --sdf-dir "$output_dir/sampled" \
    --save-dir "$output_dir" \
    --pdbqt-dir ./your_pdbqt_folder \
    --pdb-dir ./your_pdb_file \
    --dataset pdb_file \
    --write-csv \
    --write-dict \
    --docking-mode vina_dock
