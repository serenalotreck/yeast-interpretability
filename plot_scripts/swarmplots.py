"""
Script to create swarmplots for independent local model interpretation.

Author: Serena G. Lotreck, with swarmplot code adapted from
http://savvastjortjoglou.com/intrepretable-machine-learning-nfl-combine
"""
import argparse
import os

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns


def make_plot(plot_data, plot_title, y_min, y_max, out_loc):
    """
    Makes and saves one figure.

    parameters:
        plot_data, pandas df: data for one error quartile
        plot_title, str: name for this plot
        y_min, float: minimum value for y axis
        y_max, float: maximum value for y axis
        out_loc, str: place to save the plot

    returns: None
    """
    print(f'\n\nPlot being made for {plot_title}')

    # Set up axes
    fig, ax = plt.subplots()
    ax.set_ylim(y_min, y_max)

    # Create colorbar object
    cmap = sns.light_palette("seagreen", as_cmap=True)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    colors = {}
    for cval in plot_data["feat_value"]:
        colors.update({cval:cmap(norm(cval))})

    # Create the figure
    m = sns.swarmplot(x="feature", y="contrib", hue="feat_value", data=plot_data,
                        palette=colors)
    plt.xticks(rotation=45)
    m.set_title(plot_title)
    m.set_xlabel('Feature')
    m.set_ylabel('Contribution')

    # Get rid of legend to replace with colorbar
    plt.gca().legend_.remove()

    # Add colorbar
    divider = make_axes_locatable(plt.gca())
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig.add_axes(ax_cb)
    cb = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm,
                                            orientation='vertical')
    cb.set_label('Normalized feature value')

    # Make tight layout
    plt.tight_layout()

    # Save figure
    savename = re.sub(r'[^\w\s]', '', plot_title)
    savename = savename.replace(' ', '_')
    savename = savename.lower()
    print(f'\nSaving plot as {savename}.png')
    plt.savefig(f'{out_loc}/{savename}.png')


def main(plot_df_path, label_quartile, y_min, y_max, out_loc):
    """
    Reads in data and passes to make_plot.
    """
    # Get absolute paths
    plot_df_path = os.path.abspath(plot_df_path)
    out_loc = os.path.abspath(out_loc)

    # Read in data
    plot_df = pd.read_csv(plot_df_path, index_col=0)

    # Make plots
    for plot_quartile in plot_df.abs_error_bin.unique():
        plot_title = (f'Fitness {label_quartile.replace("_", " ")}, error {plot_quartile.replace("_", " ")}')
        plot_data = plot_df[plot_df['abs_error_bin'] == plot_quartile]
        make_plot(plot_data, plot_title, y_min, y_max, out_loc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Swarmplots from feature contributions')

    parser.add_argument('plot_df', type=str, help='Path to formatted plot data '
                        'output from format_plot_data.py')
    parser.add_argument('y_min', type=float, help='Minimum contrib value for all quartiles')
    parser.add_argument('y_max', type=float, help='Maximum contrib value for all quartiles')
    parser.add_argument('label_quartile', type=str, help='Name of label quartile, '
                        ' words separated by underscores')
    parser.add_argument('out_loc', type=str, help='Path to directory for output')

    args = parser.parse_args()

    main(args.plot_df, args.label_quartile, args.y_min, args.y_max, args.out_loc)
