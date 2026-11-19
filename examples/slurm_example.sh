#!/bin/bash
# Example batch script (adapt partition, cores and paths to your cluster).
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 48:00:00
#SBATCH -o %x_%j.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# RUN_DIR must contain data/{train,valid,test}.extxyz
RUN_DIR=$1; SEED=${2:-20260850}; UPDATES=${3:-5000}
python scripts/train_mace.py "$RUN_DIR" "$SEED" "$UPDATES" --device cpu "${@:4}"
