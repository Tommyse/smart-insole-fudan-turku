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

#%%
from IPython import get_ipython

#%%
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from columns import DataColumns
from dataHandler import DataHandler
from svmClassifiers import SvmClassifiers
from collections import namedtuple

from sklearn.metrics import roc_auc_score

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


#%% Classifier testing, rbf kernel
x_cols = DataColumns.getSelectedCols2()
y_cols = ["label"]
plots = True

#Parameters
kern = "rbf"

avg_acc, real_label, pred_label = SvmClassifiers.testSvm(standardized_data, kern, x_cols, y_cols, plots)

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
permutation_count = 400 #how many times the suffled data is tested

permutation_accs, permutation_aucs = SvmClassifiers.testSvmLearning(standardized_data, kern, x_cols, y_cols, permutation_count, True, avg_acc, avg_auc, "svm1_rbf")



#%%	Calculating the accuracy p-value

print("Accuracy:")
counter=0
for num in range(0,len(permutation_accs)): #going through all results
    if permutation_accs[num] >= avg_acc:
        counter = counter+1
pscore = counter/permutation_count
print("counter =",counter)
print("p-value =",pscore)


#%%	Calculating the AUC p-value

print("AUC:")
counter=0
for num in range(0,len(permutation_aucs)): #going through all results
    if permutation_aucs[num] >= avg_auc:
        counter = counter + 1
pscore = counter/permutation_count
print("counter =",counter)
print("p-value =",pscore)






#%%

# poly kernel gives bad results

#x_cols = DataColumns.getSelectedCols2()
#y_cols = ["label"]
#plots = True
#
##Parameters
#kern = "poly"
#
#avg_acc, real_label, pred_label = SvmClassifiers.testSvm(standardized_data, kern, x_cols, y_cols, plots)
#
#pred_label_df = pred_label
#real_label_df = real_label
#    
#pred_label_df = pred_label_df.replace("Normal", 0)
#pred_label_df = pred_label_df.replace("Fall", 1)
#
#real_label_df = real_label_df.replace("Normal", 0)
#real_label_df = real_label_df.replace("Fall", 1)
#
#avg_auc = roc_auc_score(real_label_df, pred_label_df)
#print("AUC score: ", round(avg_auc, 2))