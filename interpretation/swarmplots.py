"""
Script to create swarmplots for local model interpretation.

Author: Serena G. Lotreck, with code adapted from
http://savvastjortjoglou.com/intrepretable-machine-learning-nfl-combine
"""
## STEPS:
# import trained model
# determine which instances are correctly predicted and which aren't --> separate module
# make one swarmplot for all features, color by correct/incorrect
# make swarmplots for correctly and incorrectly predicted instances separately

import mispredictions as mp
