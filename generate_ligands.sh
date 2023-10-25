#!/bin/sh

for i in 0; do
    echo "${i}"
    mkdir /scratch1/e3moldiffusion/logs/crossdocked/x0_finetune/evaluation/eval_"${i}"
    python experiments/generate_ligands.py --model-path /scratch1/e3moldiffusion/logs/crossdocked/x0_finetune/run0/last-v8.ckpt --save-dir /scratch1/e3moldiffusion/logs/crossdocked/x0_finetune/evaluation/eval_"${i}/docking" --test-dir /scratch1/cremej01/data/crossdocked_noH_cutoff5/test --fix-n-nodes --skip-existing
done