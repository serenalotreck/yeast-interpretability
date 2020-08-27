"""
Script to test swarmplots.py

Author: Serena G. Lotreck
Date: 07/06/2020
"""
import unittest
from swarmplots import *
import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal
from pandas.testing import assert_series_equal

class TestTopTen(unittest.TestCase):
    """
    Tests for get_top_ten from swarmplots.py

    Since the importance file always comes from the ML_pipeline, can assume
    it will always have features ordered from greatest to least, have the
    column header 'mean_imp', and have feature names as the index.
    """
    def test_top_ten_with_less_than_ten(self):
        # Generate data
        random_interp = np.random.randint(-100, 1084, size=(50, 8))
        interp_cols = ['labels', 'prediction', 'bias'] + [f'feat{i}' for i in range(5)]
        interp_df = pd.DataFrame(random_interp, columns=interp_cols)

        random_feat = np.random.randint(-20, 56, size=(50, 6))
        feat_cols = ['labels']+[f'feat{i}' for i in range(5)]
        feature_values = pd.DataFrame(random_feat, columns=feat_cols)

        random_imp = np.random.uniform(0.0, 1.0, size=(5,1))
        index_imp = [f'feat{i}' for i in range(5)]
        imp = pd.DataFrame(random_imp, columns=['mean_imp'], index=index_imp)

        # Right answer
        new_cols_interp = ['labels', 'prediction', 'bias'] + [f'feature{i}' for i in range(5)]
        interp_df = interp_df.rename(columns=new_cols_interp)

        new_cols_feat = ['labels'] + [f'feature{i}' for i in range(5)]
        feature_values = feature_values.rename(columns=new_cols_feat)

        # Use function
        interp_func, feat_func = get_top_ten(imp, interp_df, feature_values)

        assert_frame_equal(interp_func, interp_df)
        assert_frame_equal(feat_func, feature_values)


    def test_top_ten_with_ten(self):
        # Generate data
        random_interp = np.random.randint(-100, 1084, size=(50, 13))
        interp_cols = ['labels', 'prediction', 'bias'] + [f'feat{i}' for i in range(10)]
        interp_df = pd.DataFrame(random_interp, columns=interp_cols)

        random_feat = np.random.randint(-20, 56, size=(50, 11))
        feat_cols = ['labels']+[f'feat{i}' for i in range(10)]
        feature_values = pd.DataFrame(random_feat, columns=feat_cols)

        random_imp = np.random.uniform(0.0, 1.0, size=(10,1))
        index_imp = [f'feat{i}' for i in range(10)]
        imp = pd.DataFrame(random_imp, columns=['mean_imp'], index=index_imp)

        # Right answer
        new_cols_interp = ['labels', 'prediction', 'bias'] + [f'feature{i}' for i in range(10)]
        interp_df = interp_df.rename(columns=new_cols_interp)

        new_cols_feat = ['labels'] + [f'feature{i}' for i in range(10)]
        feature_values = feature_values.rename(columns=new_cols_feat)

        # Use function
        interp_func, feat_func = get_top_ten(imp, interp_df, feature_values)

        assert_frame_equal(interp_func, interp_df)
        assert_frame_equal(feat_func, feature_values)


    def test_top_ten_with_more_than_ten(self):
        # Generate data
        random_interp = np.random.randint(-100, 1084, size=(50, 23))
        interp_cols = ['labels', 'prediction', 'bias'] + [f'feat{i}' for i in range(20)]
        interp_df = pd.DataFrame(random_interp, columns=interp_cols)

        random_feat = np.random.randint(-20, 56, size=(50, 21))
        feat_cols = ['labels']+[f'feat{i}' for i in range(20)]
        feature_values = pd.DataFrame(random_feat, columns=feat_cols)

        random_imp = np.random.uniform(0.0, 1.0, size=(20,1))
        index_imp = [f'feat{i}' for i in range(20)]
        imp = pd.DataFrame(random_imp, columns=['mean_imp'], index=index_imp)

        # Right answer
        interp_df = interp_df.drop(columns=[f'feat{i}' for i in range(11, 21, 1)])
        new_cols_interp = ['labels', 'prediction', 'bias'] + [f'feature{i}' for i in range(10)]
        interp_df = interp_df.rename(columns=new_cols_interp)

        feature_values = feature_values.drop(columns=[f'feat{i}' for i in range(11, 21, 1)])
        new_cols_feat = ['labels'] + [f'feature{i}' for i in range(10)]
        feature_values = feature_values.rename(columns=new_cols_feat)

        # Use function
        interp_func, feat_func = get_top_ten(imp, interp_df, feature_values)

        assert_frame_equal(interp_func, interp_df)
        assert_frame_equal(feat_func, feature_values)


class TestGetLabelBins(unittest.TestCase):
    pass
