"""
Script to create swarmplots for independent local model interpretation.

Author: Serena G. Lotreck, with swarmplot code adapted from
http://savvastjortjoglou.com/intrepretable-machine-learning-nfl-combine
"""
import argparse
import os

import mispredictions as mp
import pandas as pd
import numpy as np
from sklearn import preprocessing

import matplotlib.pyplot as plt
import seaborn as sns


def make_bin_plot(plot_df, plot_title, features_scaled, y_name, out_loc):
    """
    Makes one figure with error bin # of subplots.

    parameters:
        bin_df, pandas df: a subset by label ID of interp_df with bin ID's
        features_scaled, pandas df: scaled feature values
        y_name, str: Name of label column in feature_table
    """
    print(f'\n\nPlot being made for {plot_title}')

    # Get error bin names to use as subplot titles and to format plots
    error_bins = plot_df.abs_error_bin.unique()
    print(f'error_bins list for plot: {error_bins}')
    if (len(error_bins) % 2) == 0:
        col_wrap_num = 2
    else: col_wrap_num = 1

    # Make plot
    g = plot_df.groupby('feature')
    feat_value = g['feat_value'].mean()
    norm = plt.Normalize(feat_value.min(), feat_value.max())
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    myPlot = sns.catplot(x='feature', y='contrib', data=plot_df, kind='swarm',
                    hue='feat_value', col='abs_error_bin', col_wrap=col_wrap_num,
                    palette='viridis', legend=False)
    myPlot.set_xticklabels(rotation=45)
    myPlot.set_titles("Error bin: {col_name}")
    myPlot.set_axis_labels(x_var='Feature', y_var='Contribution')
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.suptitle(f'Feature Contributions for {plot_title}')
    myPlot.fig.colorbar(sm, ax=myPlot.axes.ravel().tolist(), pad=0.04, aspect=30)
    # TODO: add label for colorbar
    plt.savefig(f'{out_loc}/{bin_ID}_swarmplot.png')

    ## TODO: add command line arguments, streamline with format_plot_data.py
