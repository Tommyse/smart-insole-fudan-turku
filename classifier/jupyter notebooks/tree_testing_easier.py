# This dataset has only falling data and normal walking forward

#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'classifier\\jupyter notebooks\\insoleLib'))
	print(os.getcwd())
except:
	pass

#%%
from IPython import get_ipython

#%%
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from columns import DataColumns
from dataHandler import DataHandler
from decisionTreeClassifiers import DecisionTreeClassifiers
from collections import namedtuple

#%%
data = pd.read_csv('../tommi_test_data.csv', sep=";", header=0)
data = data.loc[data["Warning_code"] == 0]
data = data.reset_index(drop=True)
basedf = data

tforce_DF = DataHandler.calculateTotalForce(data)
step_t_DF = DataHandler.calculateStepTime(data)
#force_diff_DF = DataHandler.calculateForceDiff(data) #doesn't work currently

standardized_data = DataHandler.minmaxStandardizeForces(step_t_DF)


#%% gini tree
x_cols = DataColumns.getSelectedCols2()
y_cols = ["label"]
plots = True

parameters = namedtuple("parameters", ["class_weight", "criterion", "max_depth", "max_features",
    "max_leaf_nodes", "min_samples_leaf", "min_samples_split", "min_weight_fraction_leaf", "presort", "random_state", "splitter"])

#Parameters
params = parameters(
    	class_weight=None,
		criterion='gini',
		max_depth=5,
		max_features=None,
		max_leaf_nodes=None,
		min_samples_leaf=5,
		min_samples_split=2,
		min_weight_fraction_leaf=0.0,
		presort=False,
		random_state=123,
		splitter='best'
	)

avg_acc, real_label, pred_label = DecisionTreeClassifiers.testTreePredictions(standardized_data, params, x_cols, y_cols, plots)

#%% entropy tree
x_cols = DataColumns.getSelectedCols2()
y_cols = ["label"]
plots = True

parameters = namedtuple("parameters", ["class_weight", "criterion", "max_depth", "max_features",
    "max_leaf_nodes", "min_samples_leaf", "min_samples_split", "min_weight_fraction_leaf", "presort", "random_state", "splitter"])

#Parameters
params = parameters(
    	class_weight=None,
		criterion='entropy',
		max_depth=5,
		max_features=None,
		max_leaf_nodes=None,
		min_samples_leaf=5,
		min_samples_split=2,
		min_weight_fraction_leaf=0.0,
		presort=False,
		random_state=123,
		splitter='best'
	)

avg_acc, real_label, pred_label = DecisionTreeClassifiers.testTreePredictions(standardized_data, params, x_cols, y_cols, plots)


#%%  XGBoost (Extreme Gradient Boost trees)

x_cols = DataColumns.getSelectedCols2()
y_cols = ["label"]
plots = True

parameters = namedtuple("parameters", ["class_weight", "criterion", "max_depth", "max_features",
    "max_leaf_nodes", "min_samples_leaf", "min_samples_split", "min_weight_fraction_leaf", "presort", "random_state", "splitter"])

#Parameters (unused currently)
params = []


avg_acc, real_label, pred_label = DecisionTreeClassifiers.testXGBoostPredictions(standardized_data, params, x_cols, y_cols, plots)
