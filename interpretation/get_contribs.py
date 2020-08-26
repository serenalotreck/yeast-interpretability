"""
Script to load saved model and get joint and independent contributions for a
saved ML model.

Writes independent contributions out as a csv, and exports joint contributions
as a pickle.

TODO: see if independent contrib output is still 2D for classifiers, make
pickle save option if not

Author: Serena G. Lotreck
"""
import argparse
import os

import pandas as pd
import numpy as np
import joblib
import pickle

from treeinterpreter import treeinterpreter as ti

def prepData(frame, y_name):
    """
    Preps data for treeinterpreter.

    returns: (interp_df_half, featureNames, X)
    """
    # Put labels, ID in contribution dataframe
    interp_df_half = frame[y_name].to_frame()
    interp_df_half.set_index(frame.index)

	# Get feature names to use as col names for contributions
    featureNames = frame.columns.values.tolist()
    featureNames = featureNames[1:]

	# Drop Y to format test data for ti
    X = frame.drop([y_name], axis=1)

    return (interp_df_half, featureNames, X)

def jointContribs(feature_df, test_df, model, y_name, save, save_name):
    """
    Get joint contribs for training and test set.

    Saves predictions and biases in a csv, and uses np.save to save
    contributions. No feature names or instance ID's are saved for contribs.
    """
    for frame, set in zip([feature_df, test_df], ['training', 'test']):
        # Prep data for ti
        interp_df_half, featureNames, X = prepData(frame, y_name)

        # Call ti
        prediction, bias, contributions = ti.predict(model, X, joint_contribution=True)
        prediction = prediction.squeeze()

        # Make and save df for predictions and bias that has instance ID
        # as the index
        df_idx = frame.index
        pred_bias_df = pd.DataFrame({'prediction':prediction, 'bias':bias},
                                    index=df_idx)
        pred_bias_df.to_csv(f'{save}/{save_name}_{set}_joint_bias_and_prediction.csv')
        print('Joint bias and prediction saved! \nSnapshot of saved file:\n '
        f'{pred_bias_df.head()}')

        # Save contributions as .npy binary file
        np.save(f'{save}/{save_name}_{set}_joint_contributions.npy', contributions)
        print(f'Joint contributions for {set} saved!')

def independentContribs(feature_df, test_df, model, y_name, save, save_name):
    """
    Get independent contribs for taining and test set.

    Saves one csv file with feature names and instance labels for each set of
    data.
    """
    for frame, set in zip([feature_df, test_df], ['training', 'test']):
        # Prep data for ti
        interp_df_half, featureNames, X = prepData(frame, y_name)

        # Call ti
        prediction, bias, contributions = ti.predict(model, X)

        # Add results to contribution df
        interp_df_half['bias'] = bias.tolist()
        interp_df_half['prediction'] = prediction.flatten().tolist()

        # Make df of contributions
        contrib_df = pd.DataFrame(contributions, index = frame.index,
        					columns=featureNames)

        # Make df where columns are ID, label, bias, prediction, contributions
        local_interp_df = pd.concat([interp_df_half, contrib_df], axis=1)

        local_interp_df.to_csv(f'{save}/{save_name}_{set}_independent_contribs.csv')

        print(f'Independent contributions for {set} saved!')
        print(f'Snapshot of saved file:\n{local_interp_df.iloc[:5,:5]}')


def main(feature_matrix, feat_sep, y_name, feature_selection, test_inst, model,
        save, save_name, model_save):
    """
    Separates training and test instances and generates joint and independent
    local contributions for each instance.

    Code adapted from Christina Azodi's ML-Pipeline (https://github.com/azodichr/ML-Pipeline)

    parameters:
        feature_matrix, str: path to feature matrix
        feat_sep, str: delimiter for feature_matrix file
        y_name, str: name of label column
        feature_selection, str: path to list of features used in model
        test_inst, str: path to file containing test instances
        model, str: path to saved model
        save, str: location to save contribution output
        save_name, str: prefix for saving files
        model_save, str: identifier for whether model was pickled or saved with joblib
    """
    # Read in data
    feature_df = pd.read_csv(feature_matrix, sep=feat_sep, index_col=0,
                            engine='python')

    if os.path.isfile(feature_selection):
        with open(feature_selection) as f:
            features = f.read().strip().splitlines()
            features = [y_name] + features
            feature_df = feature_df.loc[:,features]
    args.feature_selection = os.path.abspath(args.feature_selection)

    # Load model
    if model_save.lower() in ['pickle', 'pkl', 'p']:
        with open(model) as f:
            model = pickle.load(f)
    elif model_save.lower() in ['joblib', 'j']:
        model = joblib.load(model)    args.feature_selection = os.path.abspath(args.feature_selection)


    # Split test and train instances
    with open(test_inst) as f:
        test_instances = f.read().splitlines()
    try:
        test_df = feature_df.loc[test_instances, :]
        feature_df = feature_df.drop(test_instances)
    except:
        test_instances = [int(x) for x in test_instances]
        test_df = feature_df.loc[test_instances, :]
        feature_df = feature_df.drop(test_instances)

    # Interpret results
    print('==> Calculating independent contributions <==')
    independentContribs(feature_df, test_df, model, y_name, save, save_name)
    print('\n\n==> Calculating joint contributions <==')
    jointContribs(feature_df, test_df, model, y_name, save, save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get local interpretation')
    parser.add_argument('feature_matrix', help='Path to feature matrix used to '
    'generate ML model (must be the same one)')
    parser.add_argument('test_inst', help='Path to the file created by '
    'test_set.py and used to test the model (must be the same one)')
    parser.add_argument('model', help='Path to the saved model')
    parser.add_argument('-feature_selection', help='Path to a file with a list '
    'of features used to train the model (must be the same one)', default='')
    parser.add_argument('-feat_sep', help='Delimiter for feature_matrix',
    default='\t')
    parser.add_argument('-y_name', help='Name of label column', default='Y')
    parser.add_argument('-save', help='Location to save csv and pickles',
    default='')
    parser.add_argument('-save_name', help='prefix for saved files', default='')
    parser.add_argument('-model_save', help='Name of method used to save the '
    'model. If pickled, use pickle, pkl, or p. If joblib, use joblib or j',
    default='joblib')

    args = parser.parse_args()

    args.feature_matrix = os.path.abspath(args.feature_matrix)
    args.test_inst = os.path.abspath(args.test_inst)
    args.model = os.path.abspath(args.model)
    args.feature_selection = os.path.abspath(args.feature_selection)
    args.save = os.path.abspath(args.save)


    main(args.feature_matrix, args.feat_sep, args.y_name, args.feature_selection,
    args.test_inst, args.model, args.save, args.save_name, args.model_save)
