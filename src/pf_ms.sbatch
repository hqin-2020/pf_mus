#!/bin/bash

#SBATCH --job-name=pf_ms
#SBATCH --output=pf_ms.%A_%a.out
#SBATCH --error=pf_ms.%A_%a.err
#SBATCH --time=0-8:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=16
#SBATCH --cpus-per-task=2
#SBATCH --mem=56G
#SBATCH --array=1-150
#SBATCH --constraint=fdr

module load python/anaconda-2020.02

srun python3 Run_Aso1_0_CPU_$SLURM_ARRAY_TASK_ID.py  