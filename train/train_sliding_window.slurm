#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=02:00:00
#SBATCH -A lp_stefan_r0971762_test
#SBATCH -M genius

cd $SLURM_SUBMIT_DIR

python train_sliding_window.py --windowsize $windowsize --stride $stride --maxchunks $maxchunks --model $model