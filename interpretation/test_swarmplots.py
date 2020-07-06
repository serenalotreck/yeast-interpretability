"""
Script to test swarmplots.py

Author: Serena G. Lotreck
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
    def test_get_top_ten_with_less_than_10(self):
        """
        With less than ten features: all dataframes should be returned the same,
        and gini should have all features
        """
        imp = pd.DataFrame({'mean_imp':[0.4,0.3,0.2,0.1]},index=['feature1',
                                                                'feature2',
                                                                'feature3',
                                                                'feature4'])
        interp_df = pd.DataFrame({'Y':[1,2,3,4,5], 'bias':[6,6,6,6,6],
                                    'prediction':[2,3,4,5,6],'feature1':[1,1,1,1,1],
                                    'feature2':[4,5,6,7,8],'feature3':[2,2,2,2,2],
                                    'feature4':[0,0,0,0,0]},
                                    index=[111,113,112,114,115])
        feature_values = pd.DataFrame({'Y':[1,2,3,4,5,6,7],
                                    'feature1':[1,1,1,1,1,1,1],
                                    'feature2':[4,5,6,7,8,9,10],
                                    'feature3':[2,2,2,2,2,2,2],
                                    'feature4':[0,0,0,0,0,0,0]},
                                    index=[111,112,113,114,115,116,117])
        gini_true = ['feature1','feature2','feature3','feature4']
        gini, interp_result, feature_result = get_top_ten(imp, interp_df,
                                                            feature_values)
        self.assertEqual(gini, gini_true)
        assert_frame_equal(interp_result, interp_df)
        assert_frame_equal(feature_result, feature_values)


    def test_get_top_ten_with_10(self):
        """
        With exactly ten features: all dataframes should be returned the same,
        and gini should have all features
        """
        imp = pd.DataFrame({'mean_imp':[0.1,0.04,0.02,0.01,0.001,0.0001,0.00001,
                                        0.000001,0.00000001,0.000000001]},
                                        index=['feature1','feature2','feature3',
                                                'feature4','feature5','feature6',
                                                'feature7','feature8','feature9',
                                                'feature10'])
        interp_df = pd.DataFrame({'Y':[1,2,3,4,5], 'bias':[6,6,6,6,6],
                                    'prediction':[2,3,4,5,6],'feature1':[1,1,1,1,1],
                                    'feature2':[4,5,6,7,8],'feature3':[2,2,2,2,2],
                                    'feature4':[0,0,0,0,0],'feature5':[3,3,3,3,3],
                                    'feature6':[5,5,5,5,5],'feature7':[7,6,5,4,3],
                                    'feature8':[0,9,8,7,6],'feature9':[1,2,3,4,5],
                                    'feature10':[10,10,10,10,10]},
                                    index=[111,113,112,114,115])
        feature_values = pd.DataFrame({'Y':[1,2,3,4,5,6,7],
                                    'feature1':[1,1,1,1,1,1,1],
                                    'feature2':[4,5,6,7,8,9,10],
                                    'feature3':[2,2,2,2,2,2,2],
                                    'feature4':[0,0,0,0,0,0,0],
                                    'feature5':[3,3,3,3,3,3,3],
                                    'feature6':[5,5,5,5,5,5,5],
                                    'feature7':[7,6,5,4,3,2,1],
                                    'feature8':[0,9,8,7,6,5,4],
                                    'feature9':[1,2,3,4,5,6,7],
                                    'feature10':[10,10,10,10,10,10,10]},
                                    index=[111,112,113,114,115,116,117])
        gini_true = ['feature1','feature2','feature3','feature4','feature5',
                    'feature6','feature7','feature8','feature9','feature10']

        gini, interp_result, feature_result = get_top_ten(imp, interp_df,
                                                            feature_values)
        self.assertEqual(gini, gini_true)
        assert_frame_equal(interp_result, interp_df)
        assert_frame_equal(feature_result, feature_values)

    def test_get_top_ten_with_more_than_10(self):
        """
        With more than ten, should only return the top ten features
        """
        imp = pd.DataFrame({'mean_imp':[0.1,0.04,0.02,0.01,0.001,0.0001,0.00001,
                                        0.000001,0.00000001,0.000000001,0]},
                                        index=['feature1','feature2','feature3',
                                                'feature4','feature5','feature6',
                                                'feature7','feature8','feature9',
                                                'feature10','feature11'])
        interp_df = pd.DataFrame({'Y':[1,2,3,4,5], 'bias':[6,6,6,6,6],
                                    'prediction':[2,3,4,5,6],
                                    'feature1':[1,1,1,1,1],
                                    'feature2':[4,5,6,7,8],'feature3':[2,2,2,2,2],
                                    'feature4':[0,0,0,0,0],'feature5':[3,3,3,3,3],
                                    'feature6':[5,5,5,5,5],'feature7':[7,6,5,4,3],
                                    'feature8':[0,9,8,7,6],'feature9':[1,2,3,4,5],
                                    'feature10':[10,10,10,10,10],
                                    'feature11':[11,11,11,11,11]},
                                    index=[111,113,112,114,115])
        feature_values = pd.DataFrame({'Y':[1,2,3,4,5,6,7],
                                    'feature1':[1,1,1,1,1,1,1],
                                    'feature2':[4,5,6,7,8,9,10],
                                    'feature3':[2,2,2,2,2,2,2],
                                    'feature4':[0,0,0,0,0,0,0],
                                    'feature5':[3,3,3,3,3,3,3],
                                    'feature6':[5,5,5,5,5,5,5],
                                    'feature7':[7,6,5,4,3,2,1],
                                    'feature8':[0,9,8,7,6,5,4],
                                    'feature9':[1,2,3,4,5,6,7],
                                    'feature10':[10,10,10,10,10,10,10],
                                    'feature11':[4,4,4,4,4,4,4]},
                                    index=[111,112,113,114,115,116,117])
        gini_true = ['feature1','feature2','feature3','feature4','feature5',
                    'feature6','feature7','feature8','feature9','feature10']

        gini, interp_result, feature_result = get_top_ten(imp, interp_df,
                                                            feature_values)
        # Expected results
        interp_df.drop(columns=['feature11'], inplace=True)
        feature_values.drop(columns=['feature11'], inplace=True)

        self.assertEqual(gini, gini_true)
        assert_frame_equal(interp_result, interp_df)
        assert_frame_equal(feature_result, feature_values)


class TestGetBins(unittest.TestCase):
    def test_get_bins_with_all_bins(self):
        vals = np.array([-7,-2,-1,0,0,0,1,1,1,1,2,2,2,2,2,2,2,3,3,3,4,4,4,5,5,6,8])
        vals_df = pd.DataFrame(vals,columns=['col_of_interest'])
        vals_df['random'] = np.random.randint(0, 20, vals_df.shape[0])
        vals_df['stuff'] = np.random.randint(-20, 20, vals_df.shape[0])

        binned_df = get_bins(vals_df, 'col_of_interest')

        # The right answers
        # Bin0
        idx_bin0 = binned_df.index[binned_df['col_of_interest'] == -7]
        print(f'idx_bin0: \n{idx_bin0}')
        bin0_val = list(binned_df.loc[idx_bin0, 'col_of_interest_bin_ID'])

        self.assertEqual(bin0_val, ['col_of_interest_bin0'])

        # Bin1
        idx_bin1 = binned_df.index[(binned_df['col_of_interest'] == -1) |
                                    (binned_df['col_of_interest'] == -2)]
        bin1_vals = binned_df.loc[idx_bin1, 'col_of_interest_bin_ID']
        bin1_vals = list(bin1_vals.unique())

        self.assertEqual(bin1_vals, ['col_of_interest_bin1'])

        # Bin2
        idx_bin2 = binned_df.index[(binned_df['col_of_interest'] == 0) |
                                    (binned_df['col_of_interest'] == 1)]
        bin2_vals = binned_df.loc[idx_bin2, 'col_of_interest_bin_ID']
        bin2_vals = list(bin2_vals.unique())

        self.assertEqual(bin2_vals, ['col_of_interest_bin2'])

        # Bin3
        idx_bin3 = binned_df.index[(binned_df['col_of_interest'] == 2) |
                                    (binned_df['col_of_interest'] == 3)|
                                    (binned_df['col_of_interest'] == 4)]
        bin3_vals = binned_df.loc[idx_bin3, 'col_of_interest_bin_ID']
        bin3_vals = list(bin3_vals.unique())

        self.assertEqual(bin3_vals, ['col_of_interest_bin3'])

        # Bin4
        idx_bin4 = binned_df.index[(binned_df['col_of_interest'] == 5) |
                                    (binned_df['col_of_interest'] == 6)]
        bin4_vals = binned_df.loc[idx_bin4, 'col_of_interest_bin_ID']
        bin4_vals = list(bin4_vals.unique())

        self.assertEqual(bin4_vals, ['col_of_interest_bin4'])

        # Bin5
        idx_bin5 = binned_df.index[binned_df['col_of_interest'] == 8]
        bin5_val = list(binned_df.loc[idx_bin5, 'col_of_interest_bin_ID'])

        self.assertEqual(bin5_val, ['col_of_interest_bin5'])


    def test_get_bins_with_empty_bins_0_and_5(self):
        vals = np.array([-3,1,2,3,6])
        vals_df = pd.DataFrame(vals,columns=['col_of_interest'])
        vals_df['random'] = np.random.randint(0, 20, vals_df.shape[0])
        vals_df['stuff'] = np.random.randint(-20, 20, vals_df.shape[0])

        binned_df = get_bins(vals_df, 'col_of_interest')

        # The right answers
        # Bin1
        idx_bin1 = binned_df.index[binned_df['col_of_interest'] == -3]
        bin1_val = list(binned_df.loc[idx_bin1, 'col_of_interest_bin_ID'])

        self.assertEqual(bin1_val, ['col_of_interest_bin1'])

        # Bin2
        idx_bin2 = binned_df.index[binned_df['col_of_interest'] == 1]
        bin2_val = list(binned_df.loc[idx_bin2, 'col_of_interest_bin_ID'])

        self.assertEqual(bin2_val, ['col_of_interest_bin2'])

        # Bin3
        idx_bin3 = binned_df.index[(binned_df['col_of_interest'] == 2) |
                                    (binned_df['col_of_interest'] == 3)]
        bin3_vals = binned_df.loc[idx_bin3, 'col_of_interest_bin_ID']
        bin3_vals = list(bin3_vals.unique())

        self.assertEqual(bin3_vals, ['col_of_interest_bin3'])

        # Bin4
        idx_bin4 = binned_df.index[binned_df['col_of_interest'] == 6]
        bin4_val = list(binned_df.loc[idx_bin4, 'col_of_interest_bin_ID'])

        self.assertEqual(bin4_val, ['col_of_interest_bin4'])


class TestTidyFormatting(unittest.TestCase):
    def setUp(self):
        feature_index = pd.Index([0,1,2,3], name='ID')
        self.feature_df = pd.DataFrame({'Y':[0,1,2,3],'f_1':[3,4,5,6],
                                        'f_2':[7,8,9,10]},
                                        index=feature_index)

        interp_index = pd.Index([0,1], name='ID')
        self.interp_df = pd.DataFrame({'Y':[0,1],'bias':[0.5,0.5],
                                    'prediction':[0,1],'f_1':[12,13],
                                    'f_2':[14,15], 'Y_bin_ID':['bin1','bin1'],
                                    'percent_error':[0,0],
                                    'percent_error_bin_ID':['bin1','bin2']},
                                     index=interp_index)

        single_index = pd.Index([0,0,1,1], name='ID')
        result_single_idx = pd.DataFrame({'feature_x':['f_1','f_2','f_1','f_2'],
                                        'contrib':[12,14,13,15],
                                        'error_bin_ID':['bin1','bin1','bin2',
                                                        'bin2'],
                                        'feature_y':['f_1','f_2','f_1','f_2'],
                                        'value':[3,7,4,8]},
                                        index=single_index)
        result_single_idx = result_single_idx.reset_index()
        result_single_idx['sub_idx'] = result_single_idx.groupby('ID').cumcount()
        result_multi_idx = result_single_idx.set_index(['ID',
                                                        'sub_idx'])
        self.result = result_multi_idx

    def test_make_tidy_data_final_frame(self):
        plot_df = make_tidy_data(self.interp_df, self.feature_df, 'Y')

        assert_frame_equal(plot_df, self.result)

    def test_make_tidy_data_feature_cols_aligned(self):
        """
        Test that the feature columns always align (i.e. that the values and
        contribs all correspond to the correct features)
        """
        plot_df = make_tidy_data(self.interp_df, self.feature_df, 'Y')

        # Names need to be the same in order to test if series are equal
        feature_x = plot_df['feature_x']
        feature_x = feature_x.rename('feature')
        feature_y = plot_df['feature_y']
        feature_y = feature_y.rename('feature')

        assert_series_equal(feature_x, feature_y)

if __name__ == '__main__':
    unittest.main()
