#!/bin/bash --login
#SBATCH --time=83:59:59  #walltime limit
#SBATCH --ntasks=4  #has a default of 4 nodes   #number of tasks
#SBATCH --cpus-per-task=12       #number of cores per task
#SBATCH --mem=50G        #amount of memory per node
#SBATCH --job-name DM_interp      # job name
 
export PATH=/mnt/home/lotrecks/anaconda3/bin:$PATH
source activate ML_Interpretability 

cd ~/Shiu_lab/yeast_interpretability/for_Serena

srun python ~/Shiu_lab/Pipelines/ML-Pipeline/ML_regression.py -df double_mutant_fitness_matrix_for_ML_010120.txt -test DM_test_instances_011020.txt -n 1 -n_jobs 14 -feat 040220_500 -y_name fitness -alg RF -gs True -gs_type random -gs_reps 1 -cv_num 5 -save ~/Shiu_lab/yeast_interpretability/for_Serena/DM_with_interp -treeinterp T -interp_out_loc ~/Shiu_lab/yeast_interpretability/for_Serena


scontrol show job $SLURM_JOB_ID     ### write job information to SLURM output file
js -j $SLURM_JOB_ID                 ### write resource usage to SLURM output file (powertools command)
