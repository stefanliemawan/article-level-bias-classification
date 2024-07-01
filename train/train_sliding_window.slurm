#!/bin/bash -l
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --account=intro_vsc36378


cp -r $VSC_DATA $VSC_SCRATCH
cd $VSC_SCRATCH

module --force purge
module load cluster/genius/amd
module load Python/3.10.8-GCCcore-12.2.0
module load Miniconda3/4.12.0

conda init bash
conda activate $VSC_SCRATCH/envs/jobs_env
python --version

cd vsc36378/train
pip install -r requirements.txt

python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
torchrun train_sliding_window.py
