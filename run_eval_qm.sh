#!/bin/sh

for i in 0; do
    echo "${i}"
    mkdir /scratch1/e3moldiffusion/logs/geom_qm/x0_snr_qm/evaluation/eval_"${i}"
    python experiments/run_evaluation_qm.py --model-path /scratch1/e3moldiffusion/logs/geom_qm/x0_snr_qm/best_mol_stab.ckpt --save-dir /scratch1/e3moldiffusion/logs/geom_qm/x0_snr_qm/evaluation/eval_"${i}" --save-xyz --calculate-props --batch-size 70 --ngraphs 1000 --use-energy-guidance --ckpt-energy-model /scratch1/e3moldiffusion/logs/geom/energy_training_snr/run0/best.ckpt
    #python experiments/run_evaluation.py --model-path /scratch1/e3moldiffusion/logs/geom/x0_retraining_0_exp_t_snr_t_weighting_energy_guidance/best_mol_stab.ckpt --save-dir /scratch1/e3moldiffusion/logs/geom/x0_retraining_0_exp_t_snr_t_weighting_energy_guidance/eval_"${i}" --save-xyz --calculate-energy --batch-size 60 --ngraphs 10000 --ckpt-energy-model /scratch1/e3moldiffusion/logs/geom/energy_model_snr_t_weighting/run0/best.ckpt --use-energy-guidance --guidance-scale 0.1 
done