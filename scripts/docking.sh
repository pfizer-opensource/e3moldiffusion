#!/bin/sh

out_dir="/scratch1/e3moldiffusion/logs/crossdocked/x0_snr_enamineft_cutoff5_bonds5_ep10/evaluation/docking/best"

python experiments/docking.py \
    --sdf-dir "$out_dir/sampled" \
    --save-dir "$out_dir/" \
    --pdbqt-dir /scratch1/cremej01/data/crossdocked_noH_cutoff5/test/pdbqt \
    --pdb-dir /scratch1/cremej01/data/crossdocked_pdbs
    --dataset crossdocked \
    --write-csv \
    --write-dict