#%% [markdown]
# This dataset has only falling data and normal walking forward

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
from tfClassifiers import TfClassifiers
from collections import namedtuple

plt.style.use(['ggplot'])
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.13)
plt.gcf().subplots_adjust(left=0.13)
plt.rcParams["figure.figsize"] = (14,12)
plt.ticklabel_format(style='plain', useOffset=False)

#%%
data = pd.read_csv('../tommi_test_data.csv', sep=";", header=0)
data = data.loc[data["Warning_code"] == 0]
data = data.reset_index(drop=True)
basedf = data

tforce_DF = DataHandler.calculateTotalForce(data)
step_t_DF = DataHandler.calculateStepTime(data)
#force_diff_DF = DataHandler.calculateForceDiff(data) #doesn't work currently

standardized_data = DataHandler.minmaxStandardizeForces(step_t_DF)


#%%
x_cols = DataColumns.getSelectedCols2()
y_cols = ["label"]
plots = True


avg_acc, real_label, pred_label = TfClassifiers.testNn(standardized_data, x_cols, y_cols, plots)
