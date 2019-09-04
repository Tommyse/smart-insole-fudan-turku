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

from insole.dataHandler import *


#%%
pd.set_option('display.max_columns', None)


#%%
data = pd.read_csv('../combined_with_labels.csv', sep=";", header=0)
data = data.loc[data["Warning_code"] == 0]
#data = data.drop("App_time", axis=1)

data = data.loc[data["session"] == 2]

data = data.reset_index(drop=True)
basedf = data
basedf


#%%
tforce_DF = calculateTotalForce(data)
#tforce_DF
step_t_DF = calculateStepTime(data)
#step_t_DF
force_diff_DF = calculateForceDiff(data)
#force_diff_DF

featDF_L, featDF_R, featDF_Both  = getAllFeatures(data)
featDF_Both


#%%



#%%



#%%



#%%



#%%



