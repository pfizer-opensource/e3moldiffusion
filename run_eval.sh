#!/bin/sh

for i in 1; do
    echo "${i}"
    mkdir /scratch1/e3moldiffusion/logs/geom/x0_snr_t_weighting/evaluation/eval_"${i}"
    python experiments/run_evaluation.py --model-path /scratch1/e3moldiffusion/logs/geom/x0_snr_t_weighting/best_mol_stab.ckpt --save-dir /scratch1/e3moldiffusion/logs/geom/x0_snr_t_weighting/evaluation/eval_"${i}" --save-xyz --save-traj --calculate-energy --batch-size 70 --ngraphs 1000 
done
# for i in 3 4; do
#     echo "${i}"
#     mkdir /scratch1/e3moldiffusion/logs/qm9/x0_retraining_3_snr_t_weighting_lr1e4_subset75/eval_"${i}"
#     python experiments/run_evaluation.py --model-path /scratch1/e3moldiffusion/logs/qm9/x0_retraining_3_snr_t_weighting_lr1e4_subset75/best_mol_stab.ckpt --save-dir /scratch1/e3moldiffusion/logs/qm9/x0_retraining_3_snr_t_weighting_lr1e4_subset75/eval_"${i}" --save-xyz --calculate-energy --ngraphs 10000 --batch-size 100
# done