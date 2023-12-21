#!/bin/sh

#mkdir /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_finetune_cutoff8_bonds7_addfeats/evaluation/docking/addfeats_nodes_fix

mkdir /scratch1/e3moldiffusion/logs/cdk2/docking/finetune_vary_nodes_cutoff8

python experiments/generate_ligands.py \
    --model-path /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_finetune_lr1_bs32_cutoff8/best_valid.ckpt \
    --save-dir /scratch1/e3moldiffusion/logs/cdk2/docking/finetune_vary_nodes_cutoff8 \
    --test-dir /scratch1/cremej01/data/cdk2 \
    --skip-existing \
    --num-ligands-per-pocket 10000 \
    --batch-size 40 \
    --dataset-root /scratch1/cremej01/data/crossdocked_noH_cutoff8 \
    --vary-n-nodes \
    --n-nodes-bias 10
    #--fix-n-nodes
    #--build-obabel-mol \
    #--sanitize 
    #--relax-mol \
    #--max-relax-iter 500 

