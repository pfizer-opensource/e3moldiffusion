#!/bin/sh

out_dir="/scratch1/e3moldiffusion/logs/crossdocked/x0_snr_addfeats_cutoff5_bonds7/evaluation/docking/nodes_bias_large"

python experiments/docking.py \
    --sdf-dir "$out_dir/raw" \
    --out-dir "$out_dir/processed" \
    --pdbqt-dir /scratch1/cremej01/data/crossdocked_noH_cutoff5/test/pdbqt \
    --dataset crossdocked \
    --write-csv \
    --write-dict