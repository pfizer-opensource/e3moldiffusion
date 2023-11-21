#!/bin/sh

for i in 8; do
    echo "${i}"
    mkdir /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_fienetune_lr2_cutoff5_bonds7/evaluation/eval_"${i}"
    python experiments/generate_ligands.py \
    --model-path /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_fienetune_lr2_cutoff5_bonds7/run0/last.ckpt \
    --save-dir /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_fienetune_lr2_cutoff5_bonds7/evaluation/eval_"${i}/docking" \
    --test-dir /scratch1/cremej01/data/crossdocked_noH_cutoff5/test \
    --pdbqt-dir /scratch1/cremej01/data/crossdocked_noH_cutoff5/test/pdbqt \
    --write-csv \
    --write-dict \
    --skip-existing \
    --num-ligands-per-pocket 100 \
    --batch-size 70 \
    --build-obabel-mol 
    #--fix-n-nodes \
    #--relax-mol \
    #--max-relax-iter 200 \
    #--sanitize
done
