#!/bin/sh

for i in 3 4; do
    echo "${i}"
    #mkdir /scratch1/e3moldiffusion/logs/geom/x0_retraining_0_snr_t_snr_t_weighting_addFeats_bs400/evaluation_3/eval_"${i}"
    #python experiments/run_evaluation.py --model-path /scratch1/e3moldiffusion/logs/pubchem/x0_h_pretraining_25_snr_bs400/run0/last.ckpt --save-dir /scratch1/e3moldiffusion/logs/pubchem/x0_h_pretraining_25_snr_bs400/eval_"${i}" --save-xyz --calculate-energy --batch-size 70 --ngraphs 100
    python experiments/run_evaluation.py --model-path /scratch1/e3moldiffusion/logs/geom/x0_snr_t_weighting/best_mol_stab.ckpt --save-dir /scratch1/e3moldiffusion/logs/geom/x0_snr_t_weighting/eval_"${i}" --save-xyz --calculate-energy --batch-size 70 --ngraphs 10000
    #python experiments/run_evaluation.py --model-path /scratch1/e3moldiffusion/logs/geom/x0_retraining_0_exp_t_snr_t_weighting_energy_guidance/best_mol_stab.ckpt --save-dir /scratch1/e3moldiffusion/logs/geom/x0_retraining_0_exp_t_snr_t_weighting_energy_guidance/eval_"${i}" --save-xyz --calculate-energy --batch-size 60 --ngraphs 10000 --ckpt-energy-model /scratch1/e3moldiffusion/logs/geom/energy_model_snr_t_weighting/run0/best.ckpt --use-energy-guidance --guidance-scale 0.1 
done
# for i in 3 4; do
#     echo "${i}"
#     mkdir /scratch1/e3moldiffusion/logs/qm9/x0_retraining_3_snr_t_weighting_lr1e4_subset75/eval_"${i}"
#     python experiments/run_evaluation.py --model-path /scratch1/e3moldiffusion/logs/qm9/x0_retraining_3_snr_t_weighting_lr1e4_subset75/best_mol_stab.ckpt --save-dir /scratch1/e3moldiffusion/logs/qm9/x0_retraining_3_snr_t_weighting_lr1e4_subset75/eval_"${i}" --save-xyz --calculate-energy --ngraphs 10000 --batch-size 100
# done