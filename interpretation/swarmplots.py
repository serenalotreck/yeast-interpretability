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


def make_bin_plot(bin_df, features_scaled, y_name, out_loc):
    """
    Makes one figure with error bin # of subplots.

    parameters:
        bin_df, pandas df: a subset by label ID of interp_df with bin ID's
        features_scaled, pandas df: scaled feature values
        y_name, str: Name of label column in feature_table
    """
    # Get bin ID to use as plot title
    bin_ID = bin_df[f'{y_name}_bin'].unique()[0]
    plot_title = f'label {" ".join(bin_ID.split("_"))}'
    print(f'\n\nPlot being made for {plot_title}')

    # Prep bin_df and reshape:
    bin_df = bin_df.reset_index()
    bin_df = bin_df.drop(columns=['abs_error', f'{y_name}_bin', 'bias', 'prediction', y_name])
    bin_df = pd.melt(bin_df, id_vars=['ID', 'abs_error_bin'])

    # Perform inner merge with features_scaled:
    plot_df = bin_df.merge(features_scaled, how='inner', left_on=['ID', 'variable'], right_on=['ID', 'variable'])
    plot_df = plot_df.rename(columns={'variable':'feature', 'value_x':'contrib', 'value_y':'feat_value'})

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


def make_swarmplots(interp_df, out_loc, feature_values, y_name):
    """
    Makes four matplotlib figures, each with 4 subplots. Each figure
    corresponds to one label bin, and each subplot is for an error bin within
    that label bin.


    Error bins are calculated here, within each label bin.

    parameters:
        interp_df, pandas df: interp_df with columns for bin IDs
        out_loc, str: path to save figures
        feature_values, pandas df: feature table
        y_name, str: Name of label column in feature_table
    """
    # Normalize the feature values between 0 and 1
    features_scaled = feature_values.apply(lambda x: x/x.max())
    # Prep features_scaled and reshape:
    features_scaled = features_scaled.reset_index()
    features_scaled = features_scaled.rename({'index':'ID'})
    features_scaled = features_scaled.drop(columns=[y_name])
    features_scaled = pd.melt(features_scaled, id_vars=['ID'])

    print(f'features_scaled: \n\n{features_scaled.head()}')

    # Make plots
    bins = interp_df[f'{y_name}_bin'].unique()
    for i, bin in enumerate(bins):
        bin_df = interp_df[interp_df[f'{y_name}_bin'] == bin].copy()
        # Split each label bin into error bins
        print(f'\n\n==> Separating instances in label {bin} into bins by error <==')
        bin_df = get_quartile_bins(bin_df, 'abs_error')
        print(f'\nSnapshot of dataframe with error bin column added:\n\n {bin_df.head()}')
        make_bin_plot(bin_df, features_scaled, y_name, out_loc)


def get_quartile_bins(interp_df, col_name):
    """
    Splits data into quartile bins based on label column.

    parameters:
        interp_df, pandas df: interpretation dataframe.
            If used for labels, MUST have the plot_feat
        error_bin_titles = [f'Error Bin {x}' for x in error_bins]s_dfcolumn:
            | y_name | ...
            If used for error, MUSt have the column
            | abs_error |
        col_name, str: name of the column to use for splits

    returns: interp_df with the {col_name}_bin column added
    """
    q1, q2, q3 = np.percentile(interp_df[col_name], [25, 50, 75])
    bins_label = [interp_df[col_name].min()-1,
                    q1,
                    q2,
                    q3,
                    interp_df[col_name].max()+1]

    interp_df[f'{col_name}_bin'] = pd.cut(interp_df[col_name], bins=bins_label,
                                    labels=['first_quartile', 'second_quartile',
                                            'third_quartile', 'fourth_quartile'])
    return interp_df


def rename_features(interp_df, feature_values):
    """

        error_bin_titles = [f'Error Bin {x}' for x in error_bins]
    Renames features as feature1... feature10.

    parameters:
        interp_df, pandas df: interpretation dataframe. MUST have the format:
            | y_name | bias | prediction | feature1 | ... | featureN
        feature_values, pandas df: feature matrix. MUST have the format
            | y_name | feature1 | ... | featureN

    returns:
        interp_df, pandas df: interpretation dataframe with only top ten
            features, renamed feature1 to featplot_feat
        error_bin_titles = [f'Error Bin {x}' for x in error_bins]s_dfure10
        feature_values, pandas df: feature matrix with only top ten features,
            renamed feature1 to feature10
    """
    orig_names_interp = interp_df.columns.values.tolist()
    orig_names_feat = feature_values.columns.values.tolist()

    rename_interp_dict = {}
    for i, name in enumerate(orig_names_interp):
        if i < 3:
            rename_interp_dict[name] = name
        else:
            rename_interp_dict[name] = f'feature{i-3}'

    rename_featval_dict = {}
    for i, name in enumerate(orig_names_feat):
        if i == 0:
            rename_featval_dict[name] = name
        elif i == 1:
            rename_featval_dict['NA1'] = 'NA1' ##TODO figure out what the hell these two lines are for
            rename_featval_dict['NA2'] = 'NA2'
            rename_featval_dict[name] = f'feature{i-1}'
        else:
            rename_featval_dict[name] = f'feature{i-1}'

    interp_df = interp_df.rename(columns=rename_interp_dict)
    feature_values = feature_values.rename(columns=rename_featval_dict)

    interp_convert = pd.DataFrame(list(rename_interp_dict.items()),
                                columns=['old_interp', 'new_interp'])
    feat_convert = pd.DataFrame(list(rename_featval_dict.items()),
                                columns=['old_feature_values', 'new_feature_values'])

    name_df = pd.concat([interp_convert, feat_convert], axis=1)
    name_df.to_csv('feature_name_conversion.csv', index=False)

    return interp_df, feature_values


def get_top_ten(imp, interp_df, feature_values):
    """
    Drops all features besides top ten from interp_df and feature_values,
    and renames features with shorter names. Saves a file with the top ten
    feature names and their aliases.

    parameters:
        imp, pandas df: df of gini importances
        interp_df, pandas df: interpretation dataframe. MUST have the format:
            | y_name | bias | prediction | feature1 | ... | featureN
        feature_values, pandas df: feature matrix. MUST have the format
            | y_name | feature1 | ... | featureN

    returns:
        interp_df, pandas df: interpretation dataframe
        error_bin_titles = [f'Error Bin {x}' for x in error_bins]with only top ten
            features, renamed feature1 to feature10
        feature_values, pandas df: feature matrix with only top ten # Split each label bin into error bins
    print('\n\n==> Separating instances into bins by error <==')
    interp_df = get_quartile_bins(interp_df, 'abs_error')
    print(f'\nSnapshot of dataframe with error bin column added:\n\n {interp_df.head()}')features,
            renamed feature1 to feature10
    """
    # Get top ten and drop others
    if len(imp.mean_imp) > 10:
        top_ten = imp.index.tolist()[:10]
        others = imp.index.tolist()[10:]
        interp_df = interp_df.drop(columns=others)
        feature_values = feature_values.drop(columns=others)
    else:
        top_ten = imp.index.tolist()

    print(f'\nTop ten most important features are: {top_ten}')

    # Rename features
    interp_df, feature_values = rename_features(interp_df, feature_values)

    return interp_df, feature_values


def main(interp_df, feature_values, imp, y_name, out_loc):
    """
    Generates swarmplots.
    error_bin_titles = [f'Error Bin {x}' for x in error_bins]

    parameters:
        interp_df, str: treeinterpreter output from the ML pipeline
        feature_values, df: feature table used to train the model
        imp, df: gini importances from the ML pipeline
        y_name, str: Name of label column
        out_loc, str: path to save plots
    """
    # Get the ten features to use in plots
    print('\n\n==> Getting top ten most important features <==')
    interp_df, feature_values = get_top_ten(imp, interp_df, feature_values)

    # Get distribution of the label and put in bins
    print('\n\n==> Separating instances into bins based on label <==')
    interp_df = get_quartile_bins(interp_df, y_name)
    print(f'\nSnapshot of dataframe with label bin column added:\n\n {interp_df.head()}')

    # Get error for all instances
    print('\nCalculating error for all instances...')
    interp_df['abs_error'] = interp_df[y_name] - interp_df['prediction']
    print(f'\nSnapshot of dataframe with error column added:\n\n {interp_df.head()}')

    # Make swarmplots
    print('\n\n==> Making swarmplots <==')
    make_swarmplots(interp_df, out_loc,
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
    parser.add_argument('-feature_selection', type=str, help='File with list of '
    'features used to train the model, same as ML_regression.py', default='')
    parser.add_argument('-out_loc', type=str, help='path to save the plots',
    default='')

    args = parser.parse_args()

    # Get absolute paths
    args.interp_file = os.path.abspath(args.interp_file)
    args.feature_table = os.path.abspath(args.feature_table)
    args.imp_file = os.path.abspath(args.imp_file)
    args.feature_selection = os.path.abspath(args.feature_selection)
    args.out_loc = os.path.abspath(args.out_loc)

    # Read in the data
    imp = pd.read_csv(args.imp_file, index_col=0, sep='\t', engine='python')
    interp_df = pd.read_csv(args.interp_file, index_col=0, sep=args.sep_interp, engine='python')
    feature_values = pd.read_csv(args.feature_table, index_col=0, sep=args.sep_feat, engine='python')

    # Subset selected features if applicable
    if os.path.isfile(args.feature_selection):
        with open(args.feature_selection) as f:
            features = f.read().strip().splitlines()
            features = [args.y_name] + features
            print(f'first five of features list: {features[:5]}')
            feature_values = feature_values[features]  # Chooses only the features used to train the model

    main(interp_df, feature_values, imp, args.y_name, args.out_loc)
