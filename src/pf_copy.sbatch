#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=pf_copy
#SBATCH --output=pf_copy.out
#SBATCH --error=pf_copy.err
#SBATCH --time=1-23:00:00
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Load Python modules
module load python/booth/3.8/3.8.5

srun python3 copy_final.py  