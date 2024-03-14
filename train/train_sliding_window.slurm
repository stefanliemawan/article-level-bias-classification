#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00
#SBATCH -A intro_vsc36378
#SBATCH -M genius

cd $SLURM_SUBMIT_DIR

module load Miniconda3/4.12.0
module load Python/3.10.8-GCCcore-10.3.0-bare

conda activate --yes test_env

python --version

conda install --yes pip
pip install -r requirements.txt

python train_sliding_window.py --windowsize $windowsize --stride $stride --maxchunks $maxchunks --model $model