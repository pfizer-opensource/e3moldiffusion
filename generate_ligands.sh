#!/bin/sh

for i in 0; do
    echo "${i}"
    mkdir /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_finetune_cutoff5_bonds7_addfeats/evaluation/docking_"${i}"
    python experiments/generate_ligands.py \
    --model-path /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_finetune_cutoff5_bonds7_addfeats/best_valid.ckpt \
    --save-dir /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_finetune_cutoff5_bonds7_addfeats/evaluation/docking_"${i}/qvina" \
    --test-dir /scratch1/cremej01/data/crossdocked_noH_cutoff5/test \
    --pdbqt-dir /scratch1/cremej01/data/crossdocked_noH_cutoff5/test/pdbqt \
    --write-csv \
    --write-dict \
    --skip-existing \
    --num-ligands-per-pocket 100 \
    --batch-size 70 \
    #--n-nodes-bias 5 \
    #--fix-n-nodes \
    #--build-obabel-mol \
    #--sanitize 
    #--relax-mol \
    #--max-relax-iter 200 
done
