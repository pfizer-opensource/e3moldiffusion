#!/bin/sh

for i in 0; do
    echo "${i}"
    mkdir /scratch1/e3moldiffusion/logs/crossdocked/x0_finetune/evaluation/eval_"${i}"
    python experiments/run_evaluation_ligand.py --model-path /scratch1/e3moldiffusion/logs/crossdocked/x0_finetune/run0/last-v8.ckpt --save-dir /scratch1/e3moldiffusion/logs/crossdocked/x0_finetune/evaluation/eval_"${i}" --save-xyz --calculate-energy --batch-size 20 --ngraphs 100 --use-ligand-dataset-sizes
done