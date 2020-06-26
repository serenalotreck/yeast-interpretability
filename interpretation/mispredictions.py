"""
Module to separate correctly predicted instances from incorrectly predicted
instances from a RF model.

Author: Serena G. Lotreck
"""
def calculate_error(predicted,true):
    """
    Calculate the percent error of predictions. Helper for separate().

    Parameters:
        predicted, float: predicted Y value by model
        true, float: ground truth label

    Returns: the percent error
    """
    return (abs(predicted-true)/true)*100


def define_category(error):
    """
    Use 5% error to categorize prediction as correct or mispredicted.
    Helper for separate().

    Parameters:
        error, float: the percent error

    Returns: Boolean value, T if correctly predicted (less than 5% error),
    F if incorrectly predicted ( more than 5% error)
    """
    if error <= 5:
        return True
    else: return False


def get_predictions(scores):
    """
    Get predictions and ground truth for each instance.
    Helper for separate().

    Parameters:
        scores, df: the scores output from the ML pipeline.

    Returns: a df with three columns, ID, true Y, and predicted Y.
    """
    # make new df with the relevant columns
    preds = scores[['ID','Y','rep_1']].copy()

    # rename rep_1 to Y_pred
    preds = preds.rename(columns={'rep_1':'Y_pred'})

    return preds

def separate(scores):
    """
    Separates correctly predicted instances from mispredictions.

    Parameters:
        scores, df: the scores output from the ML pipeline.

    Returns: df with 5 columns, ID (str), predicted Y (float), true Y (float),
    percent error (float), and correct/incorrect prediction (boolean, T is
    correct, F if incorrect).
    """
    # get ground truth and prediction
    preds = get_predictions(scores)

    # calculate percent error and add as a column
    preds['error'] = preds.apply(lambda x: calculate_error(x.Y_pred,x.Y),axis=1)

    # use 5% error to define correctly and incorrectly predicted instances
    preds['correct'] = preds.apply(lambda x: define_category(x.error),axis=1)

    return preds
