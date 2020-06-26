"""
Script to test swarmplots.py and mispredictions.py using the auto-mpg dataset
(https://www.kaggle.com/uciml/autompg-dataset).

First, use the ML-Pipeline (https://github.com/serenalotreck/ML-Pipeline)
to get the output files used in these scripts:

python ML_preprocess.py -df ~/GitHub/yeast-interpretability/interpretation/auto-mpg.csv -na_method median -on
ehot t -sep ',' -y_name 'mpg'

python test_set.py -df ~/GitHub/yeast-interpretability/interpretation/auto-mpg.csv_mod.txt -type r -p 0.1 -y_
name 'mpg' -sep ','

python ML_regression.py -df ~/GitHub/yeast-interpretability/interpretation/auto-mpg.csv -alg RF -y_name 'mpg' -sep ',' -test ~/GitHub/yeast-interpretability/interpretation/auto-mpg.csv_mod.txt_test.txt -gs f -n 1 -cv_num 5 -treeinterp T -gs_reps 1 -gs_type random -interp_out_loc ~/GitHub/yeast-interpretability/interpretation

Tests individual functions in mispredictions.py before testing swarmplots.py.

Author: Serena G. Lotreck
"""
import argparse
import os

import pandas as pd

import mispredictions as mp
import swarmplots as swmplt

def test_mp(scores_file):
    """
    Function to test mispredictions.py
    """
    print(f'{'='*10} Testing mispredictions.py {'='*10}')

    print(f'Snapshot of the data:\n{scores_file.head()}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Swarmplots of feature importances')

    parser.add_argument('scores_file',type=str,help='<name>_scores.txt from ML-Pipeline')
    ## TODO: add args if needed

    args = parser.parse_args()
    args.scores_file = os.path.abspath(args.scores_file)

    scores_file = pd.read_csv(args.scores_file,sep='\t')
    test_mp(scores_file)
    test_swarm(scores_file)
