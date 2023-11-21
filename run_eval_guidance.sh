#!/bin/sh

for i in 1; do
    echo "${i}"
    mkdir /scratch1/e3moldiffusion/logs/geom/x0_snr_t_weighting/polarizability/maximize_polar_scale100_steps250_relax20
    #python experiments/run_evaluation.py --model-path /scratch1/e3moldiffusion/logs/geom/x0_snr_t_weighting/best_mol_stab.ckpt --save-dir /scratch1/e3moldiffusion/logs/geom/x0_snr_t_weighting/energy/maximize_energy_scale100_steps250_relax20 --save-xyz --save-traj --calculate-props --fix-noise-and-nodes --calculate-energy --batch-size 50 --ngraphs 1000 --relax-sampling --relax-steps 20 --guidance-scale 100 --guidance-steps 250 --optimization maximize --use-energy-guidance --sample-only-valid --ckpt-energy-model /scratch1/e3moldiffusion/logs/geom/energy_training_snr/run0/best.ckpt
    python experiments/run_evaluation.py --model-path /scratch1/e3moldiffusion/logs/geom/x0_snr_t_weighting/best_mol_stab.ckpt --save-dir /scratch1/e3moldiffusion/logs/geom/x0_snr_t_weighting/polarizability/maximize_polar_scale100_steps250_relax20 --save-xyz --save-traj --calculate-props --fix-noise-and-nodes --calculate-energy --batch-size 40 --ngraphs 1000 --relax-sampling --relax-steps 20 --guidance-scale 100 --guidance-steps 250 --optimization maximize --use-energy-guidance --sample-only-valid --ckpt-energy-model /scratch1/e3moldiffusion/logs/geom_qm/x0_snr_qm/polarizability.ckpt
    #python experiments/run_evaluation.py --model-path /scratch1/e3moldiffusion/logs/geom/x0_snr_t_weighting/best_mol_stab.ckpt --save-dir /scratch1/e3moldiffusion/logs/geom/x0_snr_t_weighting/polarizability/eval_lower_pol_scale100_fix_noise_full_relax20 --save-xyz --save-traj --calculate-props --fix-noise-and-nodes --calculate-energy --batch-size 50 --ngraphs 500 --relax-sampling --relax-steps 20 --guidance-scale 1.0e-3 --use-energy-guidance --ckpt-energy-model /scratch1/e3moldiffusion/logs/geom_qm/x0_snr_qm/polarizability.ckpt 
done
# for i in 3 4; do
#     echo "${i}"
#     mkdir /scratch1/e3moldiffusion/logs/qm9/x0_retraining_3_snr_t_weighting_lr1e4_subset75/eval_"${i}"
#     python experiments/run_evaluation.py --model-path /scratch1/e3moldiffusion/logs/qm9/x0_retraining_3_snr_t_weighting_lr1e4_subset75/best_mol_stab.ckpt --save-dir /scratch1/e3moldiffusion/logs/qm9/x0_retraining_3_snr_t_weighting_lr1e4_subset75/eval_"${i}" --save-xyz --calculate-energy --ngraphs 10000 --batch-size 100
# done