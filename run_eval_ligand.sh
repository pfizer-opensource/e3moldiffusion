#!/bin/sh

for i in 0; do
    echo "${i}"
    mkdir /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_lr2_bs64/evaluation/eval_"${i}"
    python experiments/run_evaluation_ligand.py --model-path /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_lr2_bs64/run0/last.ckpt --save-dir /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_lr2_bs64/evaluation/eval_"${i}" --save-xyz --calculate-energy --batch-size 20 --ngraphs 100 --use-ligand-dataset-sizes
done