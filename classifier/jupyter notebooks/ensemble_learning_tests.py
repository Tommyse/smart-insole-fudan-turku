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

from dataHandler import DataHandler
from treeClassifiers import TreeClassifiers
from columns import DataColumns
from ensemble import Ensemble

from sklearn.metrics import classification_report

plt.style.use(['ggplot'])
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.13)
plt.gcf().subplots_adjust(left=0.13)
plt.rcParams["figure.figsize"] = (14,12)
plt.ticklabel_format(style='plain', useOffset=False)

#%%
#data = pd.read_csv('../tommi_test_data.csv', sep=";", header=0)
data = pd.read_csv('../tommi_test_data_more_diff_steps.csv', sep=";", header=0)

data = data.loc[data["Warning_code"] == 0]
data = data.reset_index(drop=True)

tforce_DF = DataHandler.calculateTotalForce(data)
step_t_DF = DataHandler.calculateStepTime(data)
#force_diff_DF = DataHandler.calculateForceDiff(data)


#%%


avg_acc, real_label_df, pred_label_df = Ensemble.testBagging(step_t_DF)










#%%
train = step_t_DF
unlabeled = step_t_DF #FOR TESTING ONLY

predictions = Ensemble.getBaggingPredictions(train, unlabeled)


print(classification_report(train["label"].values, predictions))

