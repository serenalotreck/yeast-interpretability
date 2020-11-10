"""
Script to make scatterplots of feature value  vs. contribution

Author: Serena G. Lotreck
"""
import argparse
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def make_scatterplot(feature_data, label_quartile, error_bin, out_loc):
    """
    Makes and saves a scatterplot of feat_value vs. contrib

    parameters:
        feature_data, df: subset of tidy data from format_data.py for one feature
        label_quartile, str: name of label quartile this data is from
        error_bin, str: name of error bin this data is from
        out_loc, str: path to save the plots

    returns: None
    """
    # Make names
    feature = feature_data.feature.unique()[0]
    plot_title = f'''Feature value vs. contribution for {feature}
    label {label_quartile.replace("_", " ")}
    and error {error_bin.replace("_", " ")}'''
    savename = f'{feature}_{label_quartile}_{error_bin}'
    print(f'\nPlot being made for {feature}, '
        f'label {label_quartile.replace("_", " ")} '
        f'and error {error_bin.replace("_", " ")}')

    # Make plot
    x = feature_data.feat_value
    y = feature_data.contrib

    plt.scatter(x, y, c='blue')
    plt.title(plot_title, fontsize=10)
    plt.xlabel('Feature Value')
    plt.ylabel('Contribution')
    plt.tight_layout()

    # Save plot
    print(f'Saving plot as {savename}.png...')
    plt.savefig(f'{out_loc}/{savename}.png')
    print('Saved!')

def main(plot_df, label_quartile, out_loc):
    """
    Makes scatterplots.
    """
    error_bins = plot_df.abs_error_bin.unique()

    for error_bin in error_bins:
        # Get the data in this bin
        bin_data = plot_df[plot_df['abs_error_bin'] == error_bin].copy()

        # For each feature, make a plot
        features = bin_data.feature.unique()
        for feature in features:
            feature_data = bin_data[bin_data['feature'] == feature].copy()
            make_scatterplot(feature_data, label_quartile, error_bin, out_loc)

    print('\nDone!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make scatterplots of feature '
                                    'value vs. contribution')

    parser.add_argument('plot_df', type=str, help='Path to the dataframe from '
                        'format_plot_data.py')
    parser.add_argument('label_quartile', type=str, help='Name of label quartile')
    parser.add_argument('out_loc', type=str, help='Path to save the plots')
    args = parser.parse_args()

    args.plot_df = os.path.abspath(args.plot_df)
    args.out_loc = os.path.abspath(args.out_loc)

    plot_df = pd.read_csv(args.plot_df)

    main(plot_df, args.label_quartile, args.out_loc)
