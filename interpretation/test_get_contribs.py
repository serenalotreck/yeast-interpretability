"""
Test suite for get_contribs.py

This test suite uses the output of using https://github.com/azodichr/ML-Pipeline
on the auto-mpg and iris datasets. These files will be included in the repository for
convenience, in the directories interpretation/auto-mpg-test-data/ and
interpretation/iris-test-data/

Author: Serena G. Lotreck
"""
import unittest
from get_contribs import *
import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal
from pandas.testing import assert_series_equal

class Test
