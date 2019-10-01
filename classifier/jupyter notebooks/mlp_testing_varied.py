#%% [markdown]
# This dataset has falling data and various different steps labeled as normal (forwards, sideways, backwards, etc)

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
from mlpClassifiers import MlpClassifiers
from collections import namedtuple

from sklearn.metrics import roc_auc_score

plt.style.use(['ggplot'])
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.13)
plt.gcf().subplots_adjust(left=0.13)
plt.rcParams["figure.figsize"] = (14,12)
plt.ticklabel_format(style='plain', useOffset=False)

#%% errors out from data

data = pd.read_csv('../tommi_test_data_more_diff_steps.csv', sep=";", header=0) #harder data with various stepping styles
data = data.loc[data["Warning_code"] == 0]
data = data.reset_index(drop=True)
basedf = data

tforce_DF = DataHandler.calculateTotalForce(data)
step_t_DF = DataHandler.calculateStepTime(data)
#force_diff_DF = DataHandler.calculateForceDiff(data) #doesn't work currently

standardized_data = DataHandler.minmaxStandardizeForces(step_t_DF)

#generating datasets where data has been suffled and last random x rows has been dropped and data is now balanced
dataset_amount = 1
drop_amount = 40
datasets = DataHandler.genRandomDatasets(data, dataset_amount, drop_amount)
data = datasets[0]


#%% best alpha test

x_cols = DataColumns.getSelectedCols2()
y_cols = ["label"]

mlp_parameters = namedtuple("parameters", ["hidden_layer_sizes", "solver", "alpha", "batch_size", "learning_rate",
    "learning_rate_init", "max_iter", "random_state", "verbose", "early_stopping", "validation_fraction"])

#Parameters
params = mlp_parameters(
		hidden_layer_sizes=[100, 100],
		solver="lbfgs",
		alpha=0, #ignored in this test
		batch_size="auto",
		learning_rate="constant",
		learning_rate_init=0.001,
		max_iter=200,
		random_state=123,
		verbose=True,
		early_stopping=False,
		validation_fraction=0.1
	)

alphas = np.logspace(-6, 1, 8) #alpha values to test
print("alphas: ", alphas)

best_a = MlpClassifiers.findBestAlpha(data, x_cols, y_cols, params, alphas)


#%% testing MLP

x_cols = DataColumns.getSelectedCols2()
y_cols = ["label"]
plots = True

mlp_parameters = namedtuple("parameters", ["hidden_layer_sizes", "solver", "alpha", "batch_size", "learning_rate",
    "learning_rate_init", "max_iter", "random_state", "verbose", "early_stopping", "validation_fraction"])


#Parameters
params = mlp_parameters(
		hidden_layer_sizes=[100, 100],
		solver="lbfgs",
		alpha=best_a,
		batch_size="auto",
		learning_rate="constant",
		learning_rate_init=0.001,
		max_iter=200,
		random_state=123,
		verbose=True,
		early_stopping=False,
		validation_fraction=0.1
	)

avg_acc, real_label, pred_label = MlpClassifiers.testMlp(data, params, x_cols, y_cols, plots)

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

permutation_accs, permutation_aucs = MlpClassifiers.testMlpLearning(data, params, x_cols, y_cols, permutation_count, True, avg_acc, avg_auc, "mlp_perm")


