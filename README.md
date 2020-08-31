# yeast-interpretability
Using machine learning interpretability methods to study yeast genetic interaction

## `get_contribs.py`
Use this file to get independent feature contributions for each instance. <br>

This script was written with the [Shiu Lab's ML Pipeline](https://github.com/azodichr/ML-Pipeline) in mind. There are fairly specific formatting requirements for the input to this file. <br>
* The test instances must be specified as a list of ID's, in one column of a .txt file (must be tab-separated)
* The feature selection, if applicable, must be of the same format as the test instances. This is only applicable if this feature of the ML Pipeline was used.
* The feature matrix can be a .csv or .txt, but the delimiter must be specified with the `-feat_sep` argument. The feature matrix should have the columns ID, Y/Class, and features. The default name for the label column is `'Y'`, and should be specified with the `-y_name` feature if otherwise. The ID column MUST be labeled `'ID'`
* The model can be pickled with joblib or the pickle module, and this is specified with the `-model_save` argument.

For a full list of arguments, see the script. <br>

### IMPORTANT INFO FOR RUNNING THIS SCRIPT
Due to an issue where the version of the [treeinterpreter module](https://github.com/andosa/treeinterpreter) that is installed with pip is not the most recent version, the script will crash on large multi-class models if the pip version is used. In order to use this script on any model, please do the following:
1. After cloning this repository, clone the treeinterpreter repository as well. This can be cloned to within this module, or elsewhere.
2. Move the `get_contribs.py` file into the top-level directory of the treeinterpreter repository. The structure of this repository is `/treeinterpreter/treeinterpreter/treeinterpreter.py`, so you should move the script to the first `treeinterpreter/` directory. 
3. When running the script from this directory, it should work fine. If you get an error saying `AttributeError: ti has no attribute 'predict'`, you're one level too high (outside the actual treeinterpreter repository)<br>

Please let me know if there are further issues with this. <br>

### Output
For both the test and training data, the script will output:
#### Classification:
* A .csv file for each class, with instances as rows and the feature contributions as columns.
#### Regression:
* One .csv file with instances as rows and the feature contributions as columns.
