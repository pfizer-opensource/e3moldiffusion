#!/bin/bash
#SBATCH -J drugs_processing
#SBATCH --mail-user=tuan.le@pfizer.com 
#SBATCH --mail-type=ALL
#SBATCH --partition=cpu_short
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=3G
#SBATCH --cpus-per-task=32
#SBATCH --output=/home/let55/workspace/projects/e3moldiffusion/geom/slurm_outs/dataprocessing_%j.out

cd /gpfs/workspace/users/let55/projects/e3moldiffusion/geom
source activate e3moldiffusion
python data.py --dataset qm9 --max_conformers 1000 --processes 32 --chunk_size 2048 --subchunk 256
