"""
Script to create swarmplots for independent local model interpretation.

Author: Serena G. Lotreck, with swarmplot code adapted from
http://savvastjortjoglou.com/intrepretable-machine-learning-nfl-combine
"""
import argparse
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def make_plot(plot_df, plot_title, out_loc):
    """
    Makes one figure with error bin # of subplots.

    parameters:
        plot_df, pandas df: a subset by label ID of interp_df with bin ID's
        plot_title, str: name for this plot
        out_loc, str: place to save the plot
    """
    print(f'\n\nPlot being made for {plot_title.replace("_", " ")}')

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
    plt.suptitle(f'Feature Contributions for {plot_title.replace("_", " ")}')
    myPlot.fig.colorbar(sm, ax=myPlot.axes.ravel().tolist(), pad=0.04, aspect=30)
    # TODO: add label for colorbar
    plt.savefig(f'{out_loc}/{plot_title}_swarmplot.png')


def main(plot_df_path, plot_title, out_loc):
    """
    Reads in data and passes to make_plot.
    """
    # Get absolute paths
    plot_df_path = os.path.abspath(plot_df_path)
    out_loc = os.path.abspath(out_loc)

    # Read in data
    plot_df = pd.read_csv(plot_df_path, index_col=0)

    # Make plots
    make_plot(plot_df, plot_title, out_loc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Swarmplots from feature contributions')

    parser.add_argument('plot_df', type=str, help='Path to formatted plot data '
                        'output from format_plot_data.py')
    parser.add_argument('plot_title', type=str, help='Name of plot, words separated '
			'by underscores')
    parser.add_argument('out_loc', type=str, help='Path to directory for output')

    args = parser.parse_args()

    main(args.plot_df, args.plot_title, args.out_loc)
