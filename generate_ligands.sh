#!/bin/sh

for i in 2; do
    echo "${i}"
    mkdir /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_finetune/evaluation/eval_"${i}"
    python experiments/generate_ligands.py \
    --model-path /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_finetune/best_valid.ckpt \
    --save-dir /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_finetune/evaluation/eval_"${i}/docking" \
    --test-dir /scratch1/cremej01/data/crossdocked_noH_cutoff5/test \
    --pdbqt-dir /scratch1/cremej01/data/crossdocked_noH_cutoff5/test/pdbqt \
    --write-csv \
    --write-dict \
    --skip-existing \
    --fix-n-nodes \
    --relax-mol \
    --max-relax-iter 500 \
    --sanitize
done