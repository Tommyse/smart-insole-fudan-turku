#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'classifier\\jupyter notebooks\\insoleLib'))
	print(os.getcwd())
except:
	pass


from IPython import get_ipython


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from columns import DataColumns
from dataHandler import DataHandler
from treeClassifiers import TreeClassifiers
from collections import namedtuple

from sklearn.metrics import roc_auc_score

plt.style.use(['ggplot'])
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.13)
plt.gcf().subplots_adjust(left=0.13)
plt.rcParams["figure.figsize"] = (14,12)
plt.ticklabel_format(style='plain', useOffset=False)

#%%
data = pd.read_csv('../tommi+diego_test_data.csv', sep=";", header=0) #combined dataset

data = data.loc[data["Warning_code"] == 0]
data = data.reset_index(drop=True)

data = DataHandler.calculateTotalForce(data)
data = DataHandler.calculateStepTime(data)
data = DataHandler.calculateForceValues(data)
data = DataHandler.calculatePhaseForceValues(data)


#%% gini tree
x_cols = DataColumns.getSelectedCols3()
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

avg_acc, real_label, pred_label = TreeClassifiers.testTreePredictions(data, params, x_cols, y_cols, plots)

pred_label_df = pred_label
real_label_df = real_label
    
pred_label_df = pred_label_df.replace("Normal", 0)
pred_label_df = pred_label_df.replace("Fall", 1)

real_label_df = real_label_df.replace("Normal", 0)
real_label_df = real_label_df.replace("Fall", 1)

avg_auc = roc_auc_score(real_label_df, pred_label_df)
print("AUC score: ", round(avg_auc, 2))

#%% Permutation tests

# This test should give lower average accuracy than the proper implementation
permutation_count = 100 #how many times the suffled data is tested

permutation_accs, permutation_aucs = TreeClassifiers.testTreeLearning(data, params, x_cols, y_cols, permutation_count, True, avg_acc, avg_auc, "tree2_gini_perm")


#%%
data = pd.read_csv('../tommi+diego_test_data.csv', sep=";", header=0) #combined dataset

data = data.loc[data["Warning_code"] == 0]
data = data.reset_index(drop=True)

data = DataHandler.calculateTotalForce(data)
data = DataHandler.calculateStepTime(data)
data = DataHandler.calculateForceValues(data)
data = DataHandler.calculatePhaseForceValues(data)


#%% entropy tree
x_cols = DataColumns.getSelectedCols3()
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

avg_acc, real_label, pred_label = TreeClassifiers.testTreePredictions(data, params, x_cols, y_cols, plots)

pred_label_df = pred_label
real_label_df = real_label
    
pred_label_df = pred_label_df.replace("Normal", 0)
pred_label_df = pred_label_df.replace("Fall", 1)

real_label_df = real_label_df.replace("Normal", 0)
real_label_df = real_label_df.replace("Fall", 1)

avg_auc = roc_auc_score(real_label_df, pred_label_df)
print("AUC score: ", round(avg_auc, 2))

#%% Permutation tests

# This test should give lower average accuracy than the proper implementation
permutation_count = 100 #how many times the suffled data is tested

permutation_accs, permutation_aucs = TreeClassifiers.testTreeLearning(data, params, x_cols, y_cols, permutation_count, True, avg_acc, avg_auc, "tree2_entropy_perm")


#%%
data = pd.read_csv('../tommi+diego_test_data.csv', sep=";", header=0) #combined dataset

data = data.loc[data["Warning_code"] == 0]
data = data.reset_index(drop=True)

data = DataHandler.calculateTotalForce(data)
data = DataHandler.calculateStepTime(data)
data = DataHandler.calculateForceValues(data)
data = DataHandler.calculatePhaseForceValues(data)


#%%  XGBoost (Extreme Gradient Boost trees)

x_cols = DataColumns.getSelectedCols3()
y_cols = ["label"]
plots = True

parameters = namedtuple("parameters", ["class_weight", "criterion", "max_depth", "max_features",
    "max_leaf_nodes", "min_samples_leaf", "min_samples_split", "min_weight_fraction_leaf", "presort", "random_state", "splitter"])

#Parameters (unused currently)
params = []


avg_acc, real_label, pred_label = TreeClassifiers.testXGBoostPredictions(data, params, x_cols, y_cols, plots)

pred_label_df = pred_label
real_label_df = real_label
    
pred_label_df = pred_label_df.replace("Normal", 0)
pred_label_df = pred_label_df.replace("Fall", 1)

real_label_df = real_label_df.replace("Normal", 0)
real_label_df = real_label_df.replace("Fall", 1)

avg_auc = roc_auc_score(real_label_df, pred_label_df)
print("AUC score: ", round(avg_auc, 2))


#%% Permutation tests

# This test should give lower average accuracy than the proper implementation
permutation_count = 10 #how many times the suffled data is tested

permutation_accs, permutation_aucs = TreeClassifiers.testXGBoostLearning(data, params, x_cols, y_cols, permutation_count, True, avg_acc, avg_auc, "xgboost2")
