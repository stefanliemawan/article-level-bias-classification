# article-level-bias-classification

module purge
module use /apps/leuven/rocky8/skylake/2021a/modules/all
module load worker/1.6.12-foss-2021a-wice
wsub -batch train_sliding_window.slurm -data jobs.csv