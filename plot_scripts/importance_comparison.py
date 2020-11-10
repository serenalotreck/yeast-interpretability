"""
Script to make scatterplots of gini importances from two runs of the same model
with the Shiu Lab ML Pipeline

Author: Serena G. Lotreck
"""
import argparse
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def plot_imp(imp_df, savename, out_loc):
    """
    Make scatterplot of feature importances
    """
    x = imp_df['1_x']
    y = imp_df['1_y']

    plot_title = "Feature importances in run 1 vs. run 2"

    plt.scatter(x, y, c='red')
    plt.title(plot_title, fontsize=10)
    plt.xlabel('Importance in run 1')
    plt.ylabel('Importance in run 2')
    plt.tight_layout()

    # Save plot
    print(f'Saving plot as {savename}.png...')
    plt.savefig(f'{out_loc}/{savename}_importance_comparison.png')
    print('Saved!')


def main(imp_1, imp_2, savename, out_loc):
    """
    Combine the two imp dataframes and create scatterplot.

    The imp file only contains the top 500 most important features, so if there
    is significant variation between runs, this may mean that the 500 aren't
    the same between the two files. Therefore, this procedure uses an outer
    join, and saves the features with NA in one column in a .csv for later use.
    only features in common between the two will be plotted.

    NOTE this assumes the dataframes don't have headers, so make sure to remove
    headers in the original file if they exist.
    """
    # Combine the dataframes
    imp_df = pd.merge(imp_1, imp_2, how='outer', on=0)
    print(f'Merged dataframe has {imp_df.shape[0]} rows')

    # Subset out and drop rows with NA
    nans = imp_df[imp_df["1_x"].isnull() | imp_df["1_y"].isnull()]
    print(f'There are {nans.shape[0]} rows with NaN, which means only '
            f'{imp_df.shape[0] - nans.shape[0]} features are shared between '
            'the two models.')
    imp_df = imp_df.dropna()
    print(f'Number of entities remaining after dropping NaN: {imp_df.shape[0]}')

    # Save the non-shared rows:
    print(f'\nSaving mismatched features as {out_loc}/{savename}_mismatched_important_features.csv')
    nans.to_csv(f'{out_loc}/{savename}_mismatched_important_features.csv')
    print('Saved!')

    # Make the plot
    print('\nPlotting feature importances...')
    plot_imp(imp_df, savename, out_loc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare gini feature importances ')

    parser.add_argument('imp_1', type=str, help='path to imp file for first run')
    parser.add_argument('imp_2', type=str, help='path to imp file for the second run')
    parser.add_argument('savename', type=str, help='Prefix for save name')
    parser.add_argument('out_loc', type=str, help='path to save plot and df')

    args= parser.parse_args()

    args.imp_1 = os.path.abspath(args.imp_1)
    args.imp_2 = os.path.abspath(args.imp_2)
    args.out_loc = os.path.abspath(args.out_loc)

    imp_1 = pd.read_csv(args.imp_1, header=None, sep='\t')
    imp_2 = pd.read_csv(args.imp_2, header=None, sep='\t')

    print(f'Head of imp_1:\n{imp_1.head()}')

    main(imp_1, imp_2, args.savename, args.out_loc)
