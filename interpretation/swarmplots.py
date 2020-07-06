"""
Script to create swarmplots for local model interpretation.

Operates on the treeinterpreter output from the Shiu Lab ML Pipeline.

Author: Serena G. Lotreck, with swarmplot code adapted from
http://savvastjortjoglou.com/intrepretable-machine-learning-nfl-combine
"""
# STEPS:
# 0. get top ten overall features from imp file
# 1. get distribution of labels
# 2. split into 6 bins
# 3. calculate error for all instances in each bin
# 3. within each bin, split the instances into bins by error (same scheme of SDs)
# 4. make swarmplots for each label bin (6 plots per bin, 36 total)

import argparse

import mispredictions as mp
import pandas as pd
import numpy as np
from sklearn import preprocessing

import matplotlib.pyplot as plt
import seaborn as sns


def swarmplot_with_cbar(cmap, cbar_label, ax, *args, **kwargs):
    """
    Function for making a color-gradient swarm plot.

    Adapted from
    http://savvastjortjoglou.com/intrepretable-machine-learning-nfl-combine
    """
    pass


def make_tidy_data(bin_df, features_scaled, y_name):
    """
    Reshapes and combines bin_df and features_scaled to be compatible with catplot.

    parameters:
        bin_df, pandas df: a subset by label ID of interp_df with bin ID's
        features_scaled, pandas df: scaled feature values
        y_name, str: Name of label column in feature_table

    returns:
    """
    # Get a list of the error bin of all instances
    error_bin_IDs = bin_df.percent_error_bin_ID.copy()
    error_bin_list = error_bin_IDs.unique()
    print(f'error_bin_IDs = {error_bin_IDs}')

    # Drop columns so that bin_df and features_scaled have same col names
    bin_df = bin_df.drop(columns=['Y_bin_ID', 'percent_error',
                                'percent_error_bin_ID', 'Y', 'bias', 'prediction'])
    features_scaled = features_scaled.drop(columns=[y_name])
    
    # Stack bin_df
    bin_df_stacked = pd.DataFrame(bin_df.stack(), columns=['contrib'])
    bin_df_stacked = bin_df_stacked.rename_axis(('ID', 'feature'))
    bin_df_stacked = bin_df_stacked.reset_index(level='feature')
    # Add column for error_bin_ID
    bin_df_stacked['error_bin_ID'] = np.nan
    for id in error_bin_list:
        indices = error_bin_IDs.index[error_bin_IDs == id].tolist()
        bin_df_stacked.loc[indices,'error_bin_ID'] = id
    # Make into a multiindex with numerical inner index
    bin_df_stacked = bin_df_stacked.reset_index()
    bin_df_stacked['sub_idx'] = bin_df_stacked.groupby('ID').cumcount()
    bin_df_stacked = bin_df_stacked.set_index(['ID', 'sub_idx'])

    # Stack features_scaled
    features_scaled_stacked = pd.DataFrame(features_scaled.stack(), columns=['value'])
    features_scaled_stacked = features_scaled_stacked.rename_axis(('ID',
                                                                'feature'))
    features_scaled_stacked = features_scaled_stacked.reset_index(level='feature')
    # Make into a multiindex
    features_scaled_stacked = features_scaled_stacked.reset_index()
    features_scaled_stacked['sub_idx'] = features_scaled_stacked.groupby('ID').cumcount()
    features_scaled_stacked = features_scaled_stacked.set_index(['ID', 'sub_idx'])

    # Combine over the feature column and outer index
    plot_df = bin_df_stacked.merge(features_scaled_stacked, left_index=True, right_on=['ID', 'sub_idx'])
    # plot_df = plot_df.drop(columns=['feature_y'])
    # plot_df = plot_df.rename({'feature_x':'feature'})
    print(f'\n\nSnapshot of dataframe used for plotting:\n{plot_df}')

    return plot_df


def make_bin_plot(bin_df, features_scaled, y_name):
    """
    Makes one figure with error bin # of subplots.

    parameters:
        bin_df, pandas df: a subset by label ID of interp_df with bin ID's
        features_scaled, pandas df: scaled feature values
        y_name, str: Name of label column in feature_table
    """
    # Get bin ID to use as plot title
    bin_ID = bin_df.Y_bin_ID.unique()
    print(f'\n\nPlot being made for bin {bin_ID}')

    # Reshape the data
    plot_df = make_tidy_data(bin_df, features_scaled, y_name)



def make_swarmplots(interp_df, label_bin_df, gini, out_loc, feature_values, y_name):
    """
    Makes six matplotlib figures, each with 6 subfigures. Each figure corresponds
    to one label bin, and each subplot is for an error bin within that label bin.

    parameters:
        interp_df, pandas df: full dataframe from interp_file
        label_bin_df, pandas df: interp_df with columns for bin IDs
        gini, list: ten features to include on swarmplots
        out_loc, str: path to save figures
        feature_values, pandas df: feature table
        y_name, str: Name of label column in feature_table
    """
    # Normalize the feature values between 0 and 1
    features_scaled = feature_values.apply(lambda x: x/x.max())

    # Make plots
    bins = label_bin_df.Y_bin_ID.unique()
    for i, bin in enumerate(bins):
        bin_df = label_bin_df[label_bin_df['Y_bin_ID'] == bin].copy()
        bin_df_idx = bin_df.index.values.tolist()
        make_bin_plot(bin_df, features_scaled, y_name)

def get_bins(interp_df, value_name):
    """
    Split instances into 6 bins based on the value of given column.

    Bins:
        | more than 2 SD below mean | between 1 and 2 SD below mean |
        | between 1 and 0 SD below mean | between 0 and 1 SD above mean |
        | between 1 and 2 SD above mean | more than 2 SD above mean |

    parameters:
        interp_df, pandas df: dataframe of interp_file or a subset of
        value_name, str: name of the column to use

    returns: interp_df with a column for the bin ID of each instance
    """
    # Get label column
    values = interp_df[value_name]

    # Calcualte mean and SD
    mean = values.mean()
    SD = values.std()

    # Get min and max label values
    min = values.min()
    max = values.max()

    # Get bin bounds
    bin0 = mean-2*SD
    bin1 = (mean-2*SD, mean-SD)
    bin2 = (mean-SD, mean)
    bin3 = (mean, mean+SD)
    bin4 = (mean+SD, mean+2*SD)
    bin5 = mean+2*SD
    bin_list = [bin0, bin1, bin2, bin3, bin4, bin5]

    # Get bin ID for each instance
    bin_col_dfs = []
    for i, bin in enumerate(bin_list):
        if i == 0:
            bin_col_df = interp_df[bin > interp_df[value_name]]
        elif i == 5:
            bin_col_df = interp_df[bin < interp_df[value_name]]
        else:
            bin_col_df = interp_df[(bin[0] < interp_df[value_name]) &
                                (interp_df[value_name] < bin[1])].copy()
        bin_col_df[f'{value_name}_bin_ID'] = f'{value_name}_bin{i}'
        bin_col_dfs.append(bin_col_df)

    interp_df_binned = pd.concat(bin_col_dfs)

    return interp_df_binned


def get_top_ten(imp, interp_df, feature_values):
    """
    Gets the top ten most important features form the gini importance scores file.

    parameters:
        imp, pandas df: df of gini importances
        interp_df, pandas df: interpretation dataframe
        feature_values, pandas df: feature matrix

    returns:
        top_ten: list of top ten globally important features
        interp_df, pandas df: interpretation dataframe with only top ten features
        feature_values, pandas df: feature matrix with only top ten features
    """
    if len(imp.mean_imp) > 10:
        top_ten = imp.index.tolist()[:10]
        others = imp.index.tolist()[10:]
        interp_df = interp_df.drop(columns=others)
        feature_values = feature_values.drop(columns=others)
    else:
        top_ten = imp.index.tolist()

    return top_ten, interp_df, feature_values


def main(interp_file, feature_table, imp_file, sep_interp, sep_feat,
        y_name, out_loc):
    """
    Generates swarmplots.

    parameters:
        interp_file, str: path to the file containing treeinterpreter output
            from the ML pipeline
        feature_table, str: path to the file containing the feature table
        imp_file, str: path to the file containing gini importances from the
            ML pipeline
        sep_interp, str: delimiter for interp_file
        sep_feat, str: delimiter for the feature table
        y_name, str: Name of label column in feature_table
        out_loc, str: path to save plots
    """
    # Read in the data
    imp = pd.read_csv(imp_file, sep='\t', engine='python')
    interp_df = pd.read_csv(interp_file, index_col='ID', sep=sep_interp, engine='python')
    feature_values = pd.read_csv(feature_table, index_col='ID', sep=sep_feat, engine='python')

    # Get the ten features to use in plots
    print('==> Getting top ten most important features <==')
    gini, interp_df, feature_values = get_top_ten(imp, interp_df, feature_values)
    print(f'\nThe top ten features are {gini}')

    # Get distribution of the label and put in bins
    print('\n\n==> Separating instances into bins based on label <==')
    label_bin_df = get_bins(interp_df,'Y')
    print(f'\nSnapshot of dataframe with label bin column added:\n\n {label_bin_df.head()}')

    # get error for all instances
    print('\n\n==> Calculating error for all instances <==')
    label_bin_df['percent_error'] = label_bin_df.apply(lambda x: mp.calculate_error(x.prediction,
                                x.Y), axis=1)
    print(f'\nSnapshot of dataframe with error column added:\n\n {label_bin_df.head()}')

    # Split each label bin into error bins
    print('\n\n==> Separating instances into bins by error <==')
    label_bin_df = get_bins(label_bin_df,'percent_error')
    print(f'\nSnapshot of dataframe with error bin column added:\n\n {label_bin_df.head()}')

    # Make swarmplots
    print('\n\n==> Making swarmplots <==')
    make_swarmplots(interp_df, label_bin_df, gini, out_loc,
                   feature_values, y_name)
    print('\nSwarmplots finished!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Swarmplots of feature importances')

    parser.add_argument('interp_file', type=str, help='local_contribs.csv from '
    'ML-Pipeline')
    parser.add_argument('feature_table', type=str, help='Feature table fed to '
    'ML algorithm - after preprocessing')
    parser.add_argument('imp_file', type=str, help='gini importances from ML- '
    'pipeline')
    parser.add_argument('-sep_interp', type=str, help='delimiter in interp file',
    default=',')
    parser.add_argument('-sep_feat', type=str, help='delimiter in feature file',
    default=',')
    parser.add_argument('-y_name', type=str, help='Name of label column in '
    'feature_table', default='Y')
    parser.add_argument('-out_loc', type=str, help='path to save the plots',
    default='')

    args = parser.parse_args()

    main(args.interp_file, args.feature_table, args.imp_file, args.sep_interp,
    args.sep_feat, args.y_name, args.out_loc)
