#!/bin/bash -l

#SBATCH -J drugs_coords_multiGPU
#SBATCH --mail-user=tuan.le@pfizer.com 
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu_medium
#SBATCH --constraint=weka
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:v100:3
#SBATCH --output=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/drugs_coords_multiGPU_%j.out
#SBATCH --error=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/drugs_coords_multiGPU_%j.err

# attempting to access the data directory
ls /hpfs/projects/mlcs/e3moldiffusion

cd /gpfs/workspace/users/let55/projects/e3moldiffusion/geom
source activate e3moldiffusion
echo "runnning multi-gpu experiment"

args=(
    --gpus 3
    --id 1
    --dataset drugs
    --max_num_conformers 30
    --num_workers 4
    --save_dir logs/drugs_coords
    --num_epochs 100
    --sdim 128 --vdim 32 --tdim 128 --edim 32 --rbf_dim 32 --num_layers 5
    --cutoff_local 3.0
    --cutoff_global 7.5
    --lr 5e-4
    --batch_size 256
    --local_global_model
    # --conservative
    --use_bond_features
    --use_all_atom_features
    --omit_cross_product
    --vector_aggr mean
    --schedule cosine
    --num_diffusion_timesteps 100
    --max_time 0:23:45:00
    )

python train.py "${args[@]}"
