#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=pf_mus
#SBATCH --output=pf_mus.out
#SBATCH --error=pf_mus.err
#SBATCH --time=0-3:15:00
#SBATCH --partition=standard
#SBATCH --cpus-per-task=30
#SBATCH --mem=32G
#SBATCH --array=1-150

# Load Python modules
module load python/booth/3.8/3.8.5

scratch_dir="/scratch/${SLURM_JOB_USER}/pf_ms/"
mkdir -p $scratch_dir

srun python3 Run_Aso1_0_CPU_$SLURM_ARRAY_TASK_ID.py  