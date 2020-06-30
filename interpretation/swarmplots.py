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


def make_bin_plot(i, label_bin_dict, label_bin, features_scaled,
                interp_df):
    """
    Makes one figure with error bin # of subplots.

    parameters:
        i, int: number of the label bin of this figure
        label_bin_dict, dict: error bins indices for this label
        label_bin, str: ID of the bin being plotted
        features_scaled, df: scaled feature values
        interp_df, df: interp_file df
    """



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
    features_scaled = pd.DataFrame(vals_scaled, columns=cols)
    print(features_scaled.head())

    # Add bin ID's as columns on interp_df


    # Make plots
    for i, label_bin in enumerate(label_error_bin_indices):
        make_bin_plot(i, label_error_bin_indices[label_bin], label_bin, features_scaled,
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
    bin_col_dfs = []
    for i, bin in enumerate(bin_list):
        bin_col_df = interp_df[(bin[0] < interp_df[value_name]) &
                            (interp_df[value_name] < bin[1])].copy()
        bin_col_df[f'{value_name}_bin_ID'] = f'{value_name}_bin{i}'
        bin_col_dfs.append(bin_col_df)

    interp_df_binned = pd.concat(bin_col_dfs)

    return interp_df_binned


def get_top_ten(imp_file, sep_imp):
    """
    Gets the top ten most important features form the gini importance scores file.

    parameters:
        imp_file, str: path to the gini importance
        sep, str: delimiter for imp_file file

    returns: list of the names of the top ten globally important features
    """
    imp = pd.read_csv(imp_file, sep=sep_imp, engine='python')

    if len(imp.mean_imp) > 10:
        top_ten = imp.index.tolist()[:10]
    else:
        top_ten = imp.index.tolist()

    return top_ten


def main(interp_file, feature_table, imp_file, sep_interp, sep_feat, sep_imp,
        out_loc):
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
    print('==> Getting top ten most important features <==')
    gini = get_top_ten(imp_file, sep_imp)
    print(f'\nThe top ten features are {gini}')

    # Get distribution of the label and put in bins
    print('\n\n==> Separating instances into bins based on label <==')
    interp_df = pd.read_csv(interp_file, index_col='ID', sep=sep_interp, engine='python')
    label_bin_df = get_bins(interp_df,'Y')
    print(f'\nSnapshot of dataframe with label bin column added: {label_bin_df.head()}')

    # get error for all instances
    print('\n\n==> Calculating error for all instances <==')
    label_bin_df['percent_error'] = label_bin_df.apply(lambda x: mp.calculate_error(x.prediction,
                                x.Y), axis=1)
    print(f'\nSnapshot of dataframe with error column added: {label_bin_df.head()}')

    # Split each label bin into error bins
    print('\n\n==> Separating instances into bins by error <==')
    label_bin_df = get_bins(label_bin_df,'percent_error')
    print(f'\nSnapshot of dataframe with error bin column added: {label_bin_df.head()}')

    # Make swarmplots
    # print('\n\n==> Making swarmplots <==')
    # feature_values = pd.read_csv(feature_table, sep=sep_feat, engine='python')
    # print(feature_values.head())
    # make_swarmplots(interp_df, label_error_bin_indices, gini, out_loc,
    #                 feature_values)
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
    parser.add_argument('-sep_imp', type=str, help='delimiter in imp file',
    default=',')
    parser.add_argument('-out_loc', type=str, help='path to save the plots',
    default='')

    args = parser.parse_args()

    main(args.interp_file, args.feature_table, args.imp_file, args.sep_interp,
    args.sep_feat, args.sep_imp, args.out_loc)
