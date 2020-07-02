"""
Script to test swarmplots.py

Author: Serena G. Lotreck
"""
import unittest
from swarmplots import get_top_ten
from swarmplots import get_bins
from swarmplots import make_tidy_data

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
        self.assertEqual(interp_result, interp_df)
        self.assertEqual(feature_result, feature_values)


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
        self.assertEqual(interp_result, interp_df)
        self.assertEqual(feature_result, feature_values)

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
        self.assertEqual(interp_result, interp_df)
        self.assertEqual(feature_result, feature_values)


class TestGetBins(unittest.TestCase):
    def test_get_bins_with_all_bins(self):
        pass

    def test_get_bins_with_some_empty_bins(self):
        pass

class TestTidyFormatting(unittest.TestCase):
    def test_make_tidy_data(self):
        pass

if __name__ == '__main__':
    unittest.main()
