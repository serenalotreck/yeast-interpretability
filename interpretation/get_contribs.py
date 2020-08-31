"""
Script to load saved model and get joint and independent contributions for a
saved tree-based model.

Writes independent contributions out as a csv, and exports joint contributions
as a pickle.

Author: Serena G. Lotreck
"""
import argparse
import os

import pandas as pd
import numpy as np
import joblib
import pickle

from treeinterpreter import treeinterpreter as ti
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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
    featureNames.remove(y_name)

	# Drop Y to format test data for ti
    X = frame.drop([y_name], axis=1)

    return (interp_df_half, featureNames, X)

# TODO: incorporate classification for joint contribs and add back in
# def jointContribs(feature_df, test_df, model, y_name, save, save_name):
#     """
#     Get joint contribs for training and test set.
#
#     Saves predictions and biases in a csv, and uses np.save to save
#     contributions. No feature names or instance ID's are saved for contribs.
#     """
#     for frame, set in zip([feature_df, test_df], ['training', 'test']):
#         # Prep data for ti
#         interp_df_half, featureNames, X = prepData(frame, y_name)
#
#         # Call ti
#         prediction, bias, contributions = ti.predict(model, X, joint_contribution=True)
#         prediction = prediction.squeeze()
#
#         # Make and save df for predictions and bias that has instance ID
#         # as the index
#         df_idx = frame.index
#         pred_bias_df = pd.DataFrame({'prediction':prediction, 'bias':bias},
#                                     index=df_idx)
#         pred_bias_df.to_csv(f'{save}/{save_name}_{set}_joint_bias_and_prediction.csv')
#         print('Joint bias and prediction saved! \nSnapshot of saved file:\n '
#         f'{pred_bias_df.head()}')
#
#         # Save contributions as .npy binary file
#         np.save(f'{save}/{save_name}_{set}_joint_contributions.npy', contributions)
#         print(f'Joint contributions for {set} saved!')


def deconvolute_reg_contribs(interp_df_half, bias, prediction, contributions,
                            frame, featureNames, save, save_name, set):
    """
    Mkae a dataframe for treeinterpreter output.
    """
    # Add results to contribution df
    interp_df_half['bias'] = bias.tolist()
    interp_df_half['prediction'] = prediction.flatten().tolist()

    # Make df of contributions
    contrib_df = pd.DataFrame(contributions, index = frame.index,
    					columns=featureNames)

    # Make df where columns are ID, label, bias, prediction, contributions
    local_interp_df = pd.concat([interp_df_half, contrib_df], axis=1)

    local_interp_df.to_csv(f'{save}/{save_name}_{set}_reg_independent_contribs.csv')

    print(f'\nIndependent contributions for {set} saved!')
    print(f'\nSnapshot of saved file:\n{local_interp_df.iloc[:5,:5]}')


def deconvolute_clf_contribs(model, interp_df_half, featureNames, bias,
                            prediction, contributions, save, save_name, set):
    """
    Make a dataframe for each class from the output of treeinterpreter.
    """
    classes = [f'class_{i}' for i in range(model.n_classes_)]

    prediction_df = pd.DataFrame(prediction, columns=classes)
    bias_df = pd.DataFrame(bias, columns=classes)

    class_dfs = {}
    for name in range(model.n_classes_):
        # Initialize a dataframe for the class containing ID, true class, bias and predictions
        class_df = pd.DataFrame({'bias':bias_df[f'class_{name}'],
                                'class_proba_prediction':prediction_df[f'class_{name}']})
        class_df = class_df.set_index(interp_df_half.index)
        class_df = pd.concat([interp_df_half, class_df], axis=1)

        # Make a contribution dataframe for each class
        contrib_df = 0
        contrib_rows = []
        for ix in range(contributions.shape[0]):
            if ix == 0:
                inst_df = pd.DataFrame(contributions[ix],
                                        index=featureNames,
                                        columns=classes)
                contrib_df = inst_df[f'class_{name}'].to_frame().T
            else:
                inst_df = pd.DataFrame(contributions[ix],
                                        index=featureNames,
                                        columns=classes)
                contrib_rows.append(inst_df[f'class_{name}'].to_frame().T)

        # Combine contributions for all instances into a dataframe
        contrib_df = pd.concat([contrib_df]+contrib_rows, ignore_index=True)
        contrib_df = contrib_df.set_index(class_df.index)

        # Add to the overall class dataframe
        class_df = pd.concat([class_df, contrib_df], axis=1)

        # Save class dataframe
        class_df.to_csv(f'{save}/{save_name}_{set}_clf_class_{name}_independent_contribs.csv')
        print(f'\nIndependent contributions for {set} class {name} saved!')
        print(f'\nSnapshot of saved file:\n{class_df.iloc[:5,:5]}')


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

        if isinstance(model, RandomForestClassifier):
            deconvolute_clf_contribs(model, interp_df_half, featureNames, bias,
                                        prediction, contributions, save, save_name, set)

        elif isinstance(model, RandomForestRegressor):
            deconvolute_reg_contribs(interp_df_half, bias, prediction, contributions,
                                        frame, featureNames, save, save_name, set)

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
        model = joblib.load(model)
    
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
    print('\n\n==> Calculating independent contributions <==')
    independentContribs(feature_df, test_df, model, y_name, save, save_name)
    # print('\n\n==> Calculating joint contributions <==')
    # jointContribs(feature_df, test_df, model, y_name, save, save_name)


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
