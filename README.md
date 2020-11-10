# yeast-interpretability
Using machine learning interpretability methods to study yeast genetic interaction

## `get_contribs.py`
Use this file to get independent feature contributions for each instance. <br>

### Formatting Requirements
This script was written with the [Shiu Lab's ML Pipeline](https://github.com/azodichr/ML-Pipeline) in mind. There are fairly specific formatting requirements for the input to this file. <br>
* The test instances must be specified as a list of ID's, in one column of a .txt file (must be tab-separated)
* The feature selection, if applicable, must be of the same format as the test instances. This is only applicable if this feature of the ML Pipeline was used.
* The feature matrix can be a .csv or .txt, but the delimiter must be specified with the `-feat_sep` argument. The feature matrix should have the columns ID, Y/Class, and features. The default name for the label column is `'Y'`, and should be specified with the `-y_name` feature if otherwise. The ID column MUST be labeled `'ID'`
* The model can be pickled with joblib or the pickle module, and this is specified with the `-model_save` argument.

For a full list of arguments, run `python get_contribs.py -h`. <br>

### IMPORTANT INFO FOR RUNNING THIS SCRIPT
Due to an issue where the version of the [treeinterpreter module](https://github.com/andosa/treeinterpreter) that is installed with pip is not the most recent version, the script will crash on large multi-class models if the pip version is used. In order to use this script on any model, please do the following:
1. After cloning this repository, clone the treeinterpreter repository as well. This can be cloned to within this module, or elsewhere.
2. Move the `get_contribs.py` file into the top-level directory of the treeinterpreter repository. The structure of this repository is `/treeinterpreter/treeinterpreter/treeinterpreter.py`, so you should move the script to the first `treeinterpreter/` directory.
3. When running the script from this directory, it should work fine. If you get an error saying `AttributeError: ti has no attribute 'predict'`, you're one level too high (outside the actual treeinterpreter repository)<br>

Please let me know if there are further issues with this by opening an issue. <br>

### Example Usage
Example usage with the [auto-mpg dataset](https://www.kaggle.com/uciml/autompg-dataset):
```
python get_contribs.py auto-mpg.csv auto-mpg.csv_mod.txt_test.txt auto-mp
g.csv_RF.joblib -feat_sep ',' -y_name mpg -save ../auto-mpg_data/ -save_name auto-mpg
```
Fie names from the second and third arguments are a result of using the Shiu Lab ML-Pipeline

### Output
For both the test and training data, the script will output:
#### Classification:
* A .csv file for each class, with instances as rows and the feature contributions as columns.
#### Regression:
* One .csv file with instances as rows and the feature contributions as columns.

## `plot_scripts`
This directory contains scripts for formatting data and plotting.

### `format_plot_data.py`
This script:
* Selects only top ten gini important features
* Puts contribution and feature value data into tidy format to be used downstream with plotting scripts
* Normalizes the feature values between 0 and 1
* Saves a file with the max and min contribution values for use downstream.
* Saves a file with the absolute min and max feature contributions across all quartiles to use as input arguments to `swarmplots.py`
**NOTE:** A separate dataframe is made for each of the four quartiles of the label. In the auto-mpg example, it would separate the `mpg` into four quartiles and a separate file would be saved for each.

#### Example Usage
```
python format_plot_data.py auto-mpg_training_reg_independent_contribs.csv auto-mpg.csv_mod
.txt auto-mpg.csv_RF_imp -y_name mpg -out_loc ../plots/train
```

### `swarmplots.py`
Generates 4 swarmplots for the local contributions of the top 10 features, one for each quartile of the absolute error of prediction.
#### Example Usage
```
python swarmplots.py label_first_quartile_data.csv -3.0743009482992094 4.462339726292108 first_quartile ../pl
ots/train
```

### `scatterplots.py`
Plots feature value vs. contribution for top 10 features in all quartile subsets of the data (e.g. label quartile 1 & error quartile 3 & feature3)
#### Example Usage
```
python scatterplots.py label_first_quartile_data.csv first_quartile ../plots/train/s
catterplots/
```
