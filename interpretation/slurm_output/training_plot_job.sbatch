#!/bin/bash --login
#SBATCH --time=3:59:59  #walltime limit
#SBATCH --ntasks=4  #has a default of 4 nodes   #number of tasks
#SBATCH --cpus-per-task=12       #number of cores per task
#SBATCH --mem=50G        #amount of memory per node
#SBATCH --job-name training_plots      # job name

export PATH=/mnt/home/lotrecks/anaconda3/bin:$PATH
source activate ML_Interpretability

cd ~/Shiu_lab/yeast_interpretability/yeast-interpretability/interpretation

srun python swarmplots.py ../../for_Serena/training_independent_contribs.csv ../../for_Serena/double_mutant_fitness_matrix_for_ML_010120.txt ../../for_Serena/DM_with_save_imp -sep_feat '\t' -y_name 'fitness' -feature_selection ../../for_Serena/040220_500 -out_loc plots/training/


scontrol show job $SLURM_JOB_ID     ### write job information to SLURM output file
js -j $SLURM_JOB_ID                 ### write resource usage to SLURM output file (powertools command)
