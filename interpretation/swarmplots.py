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
import os

import mispredictions as mp
import pandas as pd
from sklearn import preprocessing


def swarmplot_with_cbar(cmap, cbar_label, *args, **kwargs):
    """
    Function for making a color-gradient swarm plot.

    Taken from
    http://savvastjortjoglou.com/intrepretable-machine-learning-nfl-combine
    """
    fig = plt.gcf()
    ax = sns.swarmplot(*args, **kwargs)
    ax.legend().remove() # remove the legend, because we want to set a colorbar instead
    plt.xticks(rotation=45)
    ## create colorbar ##
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="3%", pad=0.05)
    fig.add_axes(ax_cb)
    cb = ColorbarBase(ax_cb, cmap=cmap, orientation='vertical')
    cb.set_label(cbar_label, labelpad=10)

    return fig


def make_bin_plot(i, label_error_bin_indices[label_bin], features_scaled,
                interp_df):
    """
    Makes one figure with error bin # of subplots.

    parameters:
        i, int: number of the label bin of this figure
        label_error_bin_indices[label_bin], dict: error bins indices for this label
        features_scaled, df: scaled feature values
        interp_df, df: interp_file df
    """
    ######### TODO ###########
    # write this function! to make and save figure for one label bin

def make_swarmplots(interp_df, label_error_bin_indices, gini, out_loc, feature_values):
    """
    Makes six matplotlib figures, each with 6 subfigures. Each figure corresponds
    to one label bin, and each subplot is for an error bin within that label bin.

    parameters:
        interp_df, pandas df: full dataframe from interp_file
        label_error_bin_indices, dict of dict: indices for all bins
        gini, list: ten features to include on swarmplots
        out_loc, str: path to save figures
        feature_values, padnas df: feature table
    """
    # Normalize the feature values between 0 and 1
    vals = feature_values.values # returns a numpy array
    cols = feature_values.columns

    min_max_scaler = preprocessing.MinMaxScaler()
    vals_scaled = min_max_scaler.fit_transform(vals)
    features_scaled = pd.DataFrame(x_scaled, columns=cols)

    # Make plots
    for i, label_bin in enumerate(label_error_bin_indices):
        make_bin_plot(i, label_error_bin_indices[label_bin], features_scaled,
                        interp_df)

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

    returns: a dict where keys are bin ID's and values are list of indices in bin
    """
    # Get label column
    values = interp_df[value_name]

    # Calcualte mean and SD
    mean = values.mean()
    SD = values.std()

    # Get min and max label values
    min = values.min()
    max = values.max()

    # Get bin bounds ##TODO see if there's a better way to do this
    bin0 = (min, mean-2*SD)
    bin1 = (mean-2*SD, mean-SD)
    bin2 = (mean-SD, mean)
    bin3 = (mean, mean+SD)
    bin4 = (mean+SD, mean+2*SD)
    bin5 = (mean+2*SD, max)
    bin_list = [bin0, bin1, bin2, bin3, bin4, bin5]

    # Get indices for each bin
    bin_idx = {}
    for i, bin in enumerate(bin_list):
        idx_list = interp_df.index[bin[0] < interp_df[value_name] < bin[1]].tolist()
        bin_idx[f'{value_name}_bin{i}'] = idx_list

    return bin_idx


def get_top_ten(imp_file, sep):
    """
    Gets the top ten most important features form the gini importance scores file.

    parameters:
        imp_file, str: path to the gini importance
        sep, str: delimiter for imp_file file

    returns: list of the names of the top ten globally important features
    """
    imp = pd.read_csv(imp_file, sep=sep)

    if len(imp.mean_imp) > 10:
        top_ten = imp.Index.tolist()[:10]
    else:
        top_ten = imp.Index.tolist()

    return top_ten


def main(interp_file, feature_table, imp_file, sep, out_loc):
    """
    Generates swarmplots.

    parameters:
        interp_file, str: path to the file containing treeinterpreter output
            from the ML pipeline
        feature_table, str: path to the file containing the feature table
        imp_file, str: path to the file containing gini importances from the
            ML pipeline
        sep, str: delimiter for the above files
        out_loc, str: path to save plots
    """
    # Get the ten features to use in plots
    gini = get_top_ten(imp_file, sep)

    # Get distribution of the label and put in bins
    interp_df = pd.read_csv(interp_file, index=True, sep=sep)
    label_bin_indices = get_bins(interp_df)

    # get error for all instances
    df['error'] = interp_df.apply(lambda x: mp.calculate_error(x.prediction,
                                x.Y), axis=1)

    # Split each label bin into error bins
    label_error_bin_indices = {}
    for key in label_bin_indices:
        label_bin_df = interp_df[interp_df.index.isin(label_bin_indices[key])]
        error_bin_indices = get_bins(label_bin_df)
        label_error_bin_indices[key] = error_bin_indices

    # Make swarmplots
    feature_values = pd.read_csv(feature_table, sep=sep)
    make_swarmplots(interp_df, label_error_bin_indices, gini, out_loc,
                    feature_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Swarmplots of feature importances')

    parser.add_argument('interp_file', type=str, help='local_contribs.csv from '
    'ML-Pipeline')
    parser.add_argument('feature_table', type=str, help='Feature table fed to '
    'ML algorithm - after preprocessing')
    parser.add_argument('imp_file', type=str, help='gini importances from ML- '
    'pipeline')
    parser.add_argument('-sep', type=str, help='delimiter in imp and interp files',
    default=',')
    parser.add_argument('-out_loc', type=str, help='path to save the plots',
    default='')

    args = parser.parse_args()

    main(args.interp_file, args.feature_table, args.imp_file, args.sep, args.out_loc)
