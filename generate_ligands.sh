#!/bin/sh

for i in 3; do
    echo "${i}"
    mkdir /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_lr2_bs32_lig_pocket_edges/evaluation/eval_"${i}"
    python experiments/generate_ligands.py \
    --model-path /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_lr2_bs32_lig_pocket_edges/best_valid.ckpt \
    --save-dir /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_lr2_bs32_lig_pocket_edges/evaluation/eval_"${i}/docking" \
    --test-dir /scratch1/cremej01/data/crossdocked_noH_cutoff5/test \
    --pdbqt-dir /scratch1/cremej01/data/crossdocked_noH_cutoff5/test/pdbqt \
    --write-csv \
    --write-dict \
    --skip-existing \
    --num-ligands-per-pocket 100 \
    --batch-size 50 \
    #--fix-n-nodes \
    --relax-mol \
    --max-relax-iter 500 \
    --sanitize
done
