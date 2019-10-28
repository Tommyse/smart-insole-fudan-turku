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

from dataHandler import DataHandler
from treeClassifiers import TreeClassifiers
from columns import DataColumns
from ensemble import Ensemble

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

plt.style.use(['ggplot'])
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.13)
plt.gcf().subplots_adjust(left=0.13)
plt.rcParams["figure.figsize"] = (14,12)
plt.ticklabel_format(style='plain', useOffset=False)

#%% testing method for the website
train = pd.read_csv('../tommi+diego_test_data.csv', sep=";", header=0)
data2 = pd.read_csv('../tommi_test_data.csv', sep=";", header=0)

predictions, normal_count, fall_count = Ensemble.getBoostingPredictions(train, data2)

print(predictions)

print("normal_count:", normal_count)

print("fall_count:", fall_count)
