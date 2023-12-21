#!/bin/sh

out_dir="/scratch1/e3moldiffusion/logs/crossdocked/x0_snr_addfeats_cutoff5_bonds7/evaluation/docking/nodes_bias_large"
result_dir="$out_dir/finished"


echo "Starting ligand generation..."
# Submit the SLURM script (GPU allocation)
gpu_job_id=$(sbatch --parsable scripts/generate_ligands.sl)  # Capture job info of the GPU allocation
echo "The GPU job ID is: $gpu_job_id"

#gpu_job_id=$(echo "$gpu_job_info" | awk '{print $4}')

# Wait for the GPU job to finish - SLURM ACCOUNTING IS DISABLE, so not possible on AWS
# while true; do
#     job_state=$(sacct -j $gpu_job_id --format State --noheader)
#     if [ "$job_state" == "COMPLETED" ] || [ "$job_state" == "FAILED" ]; then
#         break
#     fi
#     sleep 10  # Check job state every 10 seconds
# done

# Wait until the completion flag file is generated
while [ ! -f "$result_dir" ]; do
    sleep 10  # Check every 10 seconds
done

# Revoke the GPU allocation
scancel $gpu_job_id


echo "Starting docking of generated ligands..."
cpu_job_id=$(sbatch --parsable scripts/docking.sl)  # Capture job info of the GPU allocation
echo "The CPU job ID is: $cpu_job_id"
sbatch scripts/docking.sl

# Revoke the CPU allocation (optional)
scancel $cpu_job_id

