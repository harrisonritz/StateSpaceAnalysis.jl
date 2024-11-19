#!/bin/bash

#SBATCH -t 03:00:00
#SBATCH --mem=56GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J ssa-ssid
#SBATCH --output logs/ssid_%A-%a.txt
#SBATCH --array=1
#SBATCH --mail-type FAIL,END
#SBATCH --mail-user your_email@your_domain.edu


#--------- setup environment ---------
# use juliaup to maintain version (set to current version)
RUN_JULIA="/home/hr0283/.julia/juliaup/julia-1.10.4+0.x64.linux.gnu/bin/julia"

module load matlab/R2023b


#--------- print info ---------
echo; echo;
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo; echo; 


#--------- fit --------- 
export JULIA_NUM_THREADS=1

# make sure things don't start at the same time
echo
echo "sleeping..."
echo
sleep ${SLURM_ARRAY_TASK_ID} 
echo
echo "waking up and starting..."
echo

# run
$RUN_JULIA --heap-size-hint=${SLURM_MEM_PER_NODE}M fit_example.jl ${SLURM_ARRAY_TASK_ID} 1 fit 1