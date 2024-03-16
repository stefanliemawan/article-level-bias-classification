#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:30:00
#SBATCH -A intro_vsc36378
#SBATCH -M genius

cp -r $VSC_DATA $VSC_SCRATCH
cd $VSC_SCRATCH

module --force purge
module load cluster/genius/amd
module load Python/3.10.8-GCCcore-12.2.0
module load Miniconda3/4.12.0

conda init bash
conda activate $VSC_SCRATCH/envs/jobs_env
python --version
pip install -r requirements.txt

cd vsc36378
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
torchrun train_sliding_window.py --windowsize $windowsize --stride $stride --maxchunks $maxchunks --model $model
