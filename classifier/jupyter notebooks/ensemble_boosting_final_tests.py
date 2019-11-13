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


#%%
data = pd.read_csv('../tommi+diego_test_data.csv', sep=";", header=0)

data = data.loc[data["Warning_code"] == 0]
data = data.reset_index(drop=True)

data = DataHandler.calculateTotalForce(data)
data = DataHandler.calculateStepTime(data)
data = DataHandler.calculateForceValues(data)
data = DataHandler.calculatePhaseForceValues(data)

pd.set_option('display.max_columns', None)
selected_data = data.loc[:, DataColumns.getSelectedCols3andY()]

selected_data

#%% Boosting test


avg_acc, real_label, pred_label = Ensemble.testBoosting(data)


pred_label_df = pred_label
real_label_df = real_label
    
pred_label_df = pred_label_df.replace("Normal", 0)
pred_label_df = pred_label_df.replace("Fall", 1)

real_label_df = real_label_df.replace("Normal", 0)
real_label_df = real_label_df.replace("Fall", 1)

avg_auc = roc_auc_score(real_label_df, pred_label_df)
print("AUC score: ", round(avg_auc, 2))


#%% 2d scatter
from sklearn.decomposition import PCA

x_cols = DataColumns.getSelectedCols3()
y_cols = ["label"]

x = data.loc[:, x_cols]
y = data.loc[:, y_cols]

data_df = pd.DataFrame()
data_df = data.loc[:, x_cols]


#PCA
pca=PCA(n_components=2, svd_solver='full')
pca.fit(data_df)
T=pca.transform(data_df)

#Dataframe from T
Tdf=pd.DataFrame(T)
Tdf.columns=["c1","c2"]

#Colors
real_label_rgb = []
for entry in real_label.values:
	if(entry == "Normal"):
		real_label_rgb.append([0.5, 0.8, 0])
	elif(entry == "Fall"):
		real_label_rgb.append([0, 0, 1])

pred_label_rgb = []
for entry in pred_label.values:
	if(entry == "Normal"):
		pred_label_rgb.append([0.5, 0.8, 0])
	elif(entry == "Fall"):
		pred_label_rgb.append([0, 0, 1])


#Plot with real labels
plt.scatter(Tdf["c1"], Tdf["c2"], c=real_label_rgb, alpha=0.7)
plt.title("PCA plot, real label colors")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.legend(["Fall"], fontsize=26, markerscale=2.5)
plt.savefig("../figs/boosting_PCA_real_labels.png", facecolor="w", bbox_inches="tight")
plt.show()

#Plot with prediction labels
plt.scatter(Tdf["c1"], Tdf["c2"], c=pred_label_rgb, alpha=0.7)
plt.title("PCA plot, prediction label colors")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.legend(["Fall"], fontsize=26, markerscale=2.5)
plt.savefig("../figs/boosting_PCA_pred_labels.png", facecolor="w", bbox_inches="tight")
plt.show()

#%% 3d scatter

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

x_cols = DataColumns.getSelectedCols3()
y_cols = ["label"]

x = data.loc[:, x_cols]
y = data.loc[:, y_cols]

data_df = pd.DataFrame()
data_df = data.loc[:, x_cols]


#PCA
pca=PCA(n_components=3, svd_solver='full')
pca.fit(data_df)
T=pca.transform(data_df)

#Dataframe from T
Tdf=pd.DataFrame(T)
Tdf.columns=["c1","c2","c3"]

#Colors
real_label_rgb = []
for entry in real_label.values:
	if(entry == "Normal"):
		real_label_rgb.append([0.5, 0.8, 0])
	elif(entry == "Fall"):
		real_label_rgb.append([0, 0, 1])

pred_label_rgb = []
for entry in pred_label.values:
	if(entry == "Normal"):
		pred_label_rgb.append([0.5, 0.8, 0])
	elif(entry == "Fall"):
		pred_label_rgb.append([0, 0, 1])


#Plot with real labels
fig = plt.figure(1)
ax = Axes3D(fig, elev=-150, azim=310)
ax.set_title("PCA 3D plot, real label colors")
ax.scatter(Tdf["c1"], Tdf["c2"], Tdf["c3"], c=real_label_rgb, alpha=1)
ax.set_xlabel("Principle Component 1")
ax.set_ylabel("Principle Component 2")
ax.set_zlabel("Principle Component 3")
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.legend(["Fall"], fontsize=26, markerscale=2.5)
plt.savefig("../figs/boosting_PCA_3d_real_labels_angle2.png", facecolor="w", bbox_inches="tight")
plt.show()

#Plot with prediction labels
fig = plt.figure(1)
ax = Axes3D(fig, elev=-150, azim=310)
ax.set_title("PCA 3D plot, prediction label colors")
ax.scatter(Tdf["c1"], Tdf["c2"], Tdf["c3"], c=pred_label_rgb, alpha=1)
ax.set_xlabel("Principle Component 1")
ax.set_ylabel("Principle Component 2")
ax.set_zlabel("Principle Component 3")
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.legend(["Fall"], fontsize=26, markerscale=2.5)
plt.savefig("../figs/boosting_PCA_3d_pred_labels_angle1.png", facecolor="w", bbox_inches="tight")
plt.show()

#More angles
#Plot with real labels
fig = plt.figure(1)
ax = Axes3D(fig, elev=-150, azim=110)
ax.set_title("PCA 3D plot, real label colors")
ax.scatter(Tdf["c1"], Tdf["c2"], Tdf["c3"], c=real_label_rgb, alpha=1)
ax.set_xlabel("Principle Component 1")
ax.set_ylabel("Principle Component 2")
ax.set_zlabel("Principle Component 3")
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.legend(["Fall"], fontsize=26, markerscale=2.5)
plt.savefig("../figs/boosting_PCA_3d_real_labels_angle2.png", facecolor="w", bbox_inches="tight")
plt.show()

#Plot with prediction labels
fig = plt.figure(1)
ax = Axes3D(fig, elev=-150, azim=110)
ax.set_title("PCA 3D plot, prediction label colors")
ax.scatter(Tdf["c1"], Tdf["c2"], Tdf["c3"], c=pred_label_rgb, alpha=1)
ax.set_xlabel("Principle Component 1")
ax.set_ylabel("Principle Component 2")
ax.set_zlabel("Principle Component 3")
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.legend(["Fall"], fontsize=26, markerscale=2.5)
plt.savefig("../figs/boosting_PCA_3d_pred_labels_angle2.png", facecolor="w", bbox_inches="tight")
plt.show()
