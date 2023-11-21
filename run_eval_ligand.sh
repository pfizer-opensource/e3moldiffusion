#!/bin/sh

for i in 1; do
    echo "${i}"
    mkdir /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_fienetune_lr2_cutoff5_bonds7/evaluation/eval_obabel_"${i}"
    #mkdir /hpfs/userws/cremej01/projects/logs/crossdocked/x0_snr_lr2_bs32_large/evaluation/eval_"${i}"
    #python experiments/run_evaluation_ligand.py --model-path /hpfs/userws/cremej01/projects/logs/crossdocked/x0_snr_lr2_bs32_large/best_valid.ckpt --save-dir /hpfs/userws/cremej01/projects/logs/crossdocked/x0_snr_lr2_bs32_large/evaluation/eval_"${i}" --save-xyz --save-traj --use-ligand-dataset-sizes --batch-size 100
    python experiments/run_evaluation_ligand.py --build-obabel-mol --model-path /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_fienetune_lr2_cutoff5_bonds7/run0/last.ckpt --save-dir /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_fienetune_lr2_cutoff5_bonds7/evaluation/eval_obabel_"${i}" --save-xyz --save-traj --batch-size 100 #--use-ligand-dataset-sizes 

done