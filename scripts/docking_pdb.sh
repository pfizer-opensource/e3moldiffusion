#!/bin/sh


mkdir /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_fienetune_lr2_cutoff5_bonds7/evaluation/eval_6/docking/qvina

python experiments/docking.py \
--pdbqt_dir /scratch1/cremej01/data/cdk2 \
--sdf_dir /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_fienetune_lr2_cutoff5_bonds7/evaluation/eval_6/docking/raw \
--out_dir /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_fienetune_lr2_cutoff5_bonds7/evaluation/eval_6/docking/qvina \
--write_csv \
--write_dict