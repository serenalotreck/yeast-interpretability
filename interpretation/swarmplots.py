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
    # Sort keys
    sorted_keys = sorted(label_bin_dict.keys())

    # Make figure
    #fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(3,2)

    # Define colorbar preferences
    cmap = plt.get_cmap('viridis')
    cbar_label = 'Feature Value Percentile'

    # Make the plot
    #ax_list = [ax1, ax2, ax3, ax4, ax5, ax6]
    for key in sorted_keys:
        # Get data for this bin
        bin_data = interp_df[interp_df.index.isin(label_bin_dict[key])]
        bin_feature_vals = features_scaled[features_scaled.index.isin(label_bin_dict[key])]

        bin_data_melted = pd.melt(bin_data, id_vars=['Y'], value_vars=bin_data.columns.values.tolist()[1:])
        print(f'Bin data melted: {bin_data_melted.head()}')
        # Make plot
        # ax = swarmplot_with_cbar(cmap, cbar_label, ax, x='feature',
        #  y='contribution', hue='scaled_feat_vals', palette='viridis',
        #   order=features, data=)


    # Make subplots
    # axes_counter = [0,0]
    # for bin in sorted_keys:
    #     # Get data for this bin
    #     bin_data = interp_df[interp_df.index.isin(label_bin_dict[bin])]
    #     bin_feature_vals = features_scaled[features_scaled.index.isin(label_bin_dict[bin])]
    #
    #     # Plot
    #     axs[axes_counter[0], axes_counter[1]].swarmplot_with_cbar(cmap, cbar_label,  x='feature', y='contribution',
    #                 hue='scaled_feat_vals', palette='viridis', order=features,
    #                 data=test_expl_df.loc[test_expl_df.feature!='']);
    #     axs[axes_counter[0], axes_counter[1]].set_title(bin)
    #
    #     # Update the axes counter
    #     if i % 2 != 0:
    #         axes_counter[0] += 1
    #
    #     axes_counter[1] = (i+1) % 2

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()
    #
    # plt.tight_layout()
    #
    # plt.savefig(label_bin+'.png')



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
    bin_idx = {}
    for i, bin in enumerate(bin_list):
        idx_df = interp_df[(bin[0] < interp_df[value_name]) &
                            (interp_df[value_name] < bin[1])]
        idx_list = idx_df.index.tolist()
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
    interp_df = pd.read_csv(interp_file, index_col='ID', sep=sep_interp)
    print(interp_df.head())
    label_bin_indices = get_bins(interp_df,'Y')
    print(f'\nBin names are {list(label_bin_indices.keys())}')

    # get error for all instances
    print('\n\n==> Calculating error for all instances <==')
    interp_df['percent_error'] = interp_df.apply(lambda x: mp.calculate_error(x.prediction,
                                x.Y), axis=1)
    print(f'\nSnapshot of dataframe with error column added: {interp_df.head()}')

    # Split each label bin into error bins
    print('\n\n==> Separating instances into bins by error <==')
    label_error_bin_indices = {}
    for key in label_bin_indices:
        label_bin_df = interp_df[interp_df.index.isin(label_bin_indices[key])]
        error_bin_indices = get_bins(label_bin_df, 'percent_error')
        label_error_bin_indices[key] = error_bin_indices
    print(f'\nError bin names are {list(label_error_bin_indices[list(label_bin_indices.keys())[0]].keys())}')

    # Make swarmplots
    print('\n\n==> Making swarmplots <==')
    feature_values = pd.read_csv(feature_table, sep=sep_feat)
    print(feature_values.head())
    make_swarmplots(interp_df, label_error_bin_indices, gini, out_loc,
                    feature_values)
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
