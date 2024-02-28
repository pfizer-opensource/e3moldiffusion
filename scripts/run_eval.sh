#!/bin/sh


python experiments/run_evaluation.py \
    --model-path /hpfs/userws/cremej01/projects/logs/geom/x0_snr/best_mol_stab.ckpt \
    --save-dir /hpfs/userws/cremej01/projects/logs/geom/x0_snr/evaluation/polar_guidance \
    --batch-size 70 \
    --ngraphs 10 \
    --ckpt-property-model /hpfs/userws/cremej01/projects/logs/geom/x0_snr/best_mol_stab.ckpt \
    --importance-sampling \
    --importance-sampling-start 0 \
    --importance-sampling-end 200 \
    --every-importance-t 5 \
    --property-tau 0.1
    # --save-xyz \
    # --save-traj \
    # --calculate-energy \

