#!/bin/bash
#SBATCH --job-name="calc_geom_qm"
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1
#SBATCH --error=arrayjob\_%j.err
#SBATCH --output=arrayjob\_%j.out
#SBATCH --time=700:00:00
#SBATCH --partition="ondemand-cpu-c48"
#SBATCH --no-requeue
#SBATCH --mem=48GB
#SBATCH --array=1-64%8


IN=/scratch1/seumej/geom_qm/raw
JOB=arrayjob

FILELIST=($(ls -d $IN/* ))

echo Looking for data files in $IN
echo Found $FILELIST

source activate e3moldiffusion

TASKFILE=${FILELIST[$SLURM_ARRAY_TASK_ID]}
echo Running on $TASKFILE

~/mambaforge/envs/e3moldiffusion/bin/python -u /sharedhome/seumej/e3moldiffusion/calc_chunk.py $TASKFILE >> array.out
