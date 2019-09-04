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
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA


from insole.dataHandler import *
from insole.decisionTreeClassifiers import *
from insole.columns import *



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
labeled = pd.read_csv("combined_with_labels.csv", sep=";", header=0)
#labeled = labeled.loc[labeled["Warning_code"] == 0]

unlabeled = pd.read_csv("combined_unlabeled_data.csv", sep=",", header=0)
#unlabeled = unlabeled.loc[unlabeled["Warning_code"] == 0]

values_cols = ["Contact_time","S0_force","S0_start_time","S0_max_time","S0_end_time","S1_force",
               "S1_start_time","S1_max_time","S2_force","S2_start_time","S2_max_time","S2_end_time",
               "S3_force","S3_start_time","S3_max_time","S3_end_time","S4_force","S4_start_time",
               "S4_max_time","S4_end_time","S5_force","S5_start_time","S5_max_time","S5_end_time",
               "S6_force","S6_start_time","S6_max_time","S6_end_time","F1_force","F1_time","F2_force",
               "F2_time","F3_force","F3_time"]

xcols = ["S0_force","S1_force","S2_force","S3_force","S4_force","S5_force","S6_force",
                "S0_press_time","S1_press_time","S2_press_time","S3_press_time","S4_press_time","S5_press_time","S6_press_time",
                "S0_force_diff","S1_force_diff","S2_force_diff","S3_force_diff","S4_force_diff","S5_force_diff","S6_force_diff","total_force_diff",
                "Contact_time","force_total"]
ycols = ["label"]


#%%
x = labeled.loc[:, values_cols] #original values in csv
y = labeled.loc[:, ycols]


unlabeled_x = unlabeled.loc[:, values_cols]
unlabeled_x


pca1=PCA(n_components=2, svd_solver='full')
pca1.fit(x)

pca2=PCA(n_components=2, svd_solver='full')
pca2.fit(unlabeled_x)

T1=pca1.transform(x)
T2=pca2.transform(unlabeled_x)

#Dataframe from T
T1df=pd.DataFrame(T1)
T1df.columns=["c1","c2"]
T2df=pd.DataFrame(T2)
T2df.columns=["c1","c2"]

colors1 = y
colors1 = y

plt.scatter(T1df["c1"], T1df["c2"], c=y)
plt.title("PCA scatter plot, labeled data")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.show()

plt.scatter(T2df["c1"], T2df["c2"], c=y)
plt.title("PCA scatter plot, unlabeled data")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.show()


#%%
#k = findBestK(x, y, 1, 10, values_cols, ycols) #testing 2 to 20
k = 3
print("Best k =",k)


#%%
semi_df = knnLabel(labeled, unlabeled_x, values_cols, ycols, k)
semi_df


#%%
semi_df_bad = semi_df.loc[semi_df["label"] == "Bad"]
x = semi_df_bad.loc[:, values_cols] #original values in csv
y = semi_df_bad.loc[:, ycols]

pca=PCA(n_components=2, svd_solver='full')
pca.fit(x)

T=pca.transform(x)

#Dataframe from T
Tdf=pd.DataFrame(T)
Tdf.columns=["c1","c2"]

plt.scatter(Tdf["c1"], Tdf["c2"], c="r")
plt.title("PCA scatter plot, Bad")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.show()

semi_df_good = semi_df.loc[semi_df["label"] == "Good"]
x = semi_df_good.loc[:, values_cols] #original values in csv
y = semi_df_good.loc[:, ycols]

pca=PCA(n_components=2, svd_solver='full')
pca.fit(x)

T=pca.transform(x)

#Dataframe from T
Tdf=pd.DataFrame(T)
Tdf.columns=["c1","c2"]

plt.scatter(Tdf["c1"], Tdf["c2"], c="b")
plt.title("PCA scatter plot, Good")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.show()


#%%
semi_df2 = labelSpreading(labeled, unlabeled_x, values_cols, ycols, alpha_v=0.8)
semi_df2


#%%
semi_df2_bad = semi_df2.loc[semi_df["label"] == "Bad"]
x = semi_df2_bad.loc[:, values_cols] #original values in csv
y = semi_df2_bad.loc[:, ycols]

pca=PCA(n_components=2, svd_solver='full')
pca.fit(x)

T=pca.transform(x)

#Dataframe from T
Tdf=pd.DataFrame(T)
Tdf.columns=["c1","c2"]

plt.scatter(Tdf["c1"], Tdf["c2"], c="r")
plt.title("PCA scatter plot, Bad")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.show()

semi_df2_good = semi_df2.loc[semi_df["label"] == "Good"]
x = semi_df2_good.loc[:, values_cols] #original values in csv
y = semi_df2_good.loc[:, ycols]

pca=PCA(n_components=2, svd_solver='full')
pca.fit(x)

T=pca.transform(x)

#Dataframe from T
Tdf=pd.DataFrame(T)
Tdf.columns=["c1","c2"]

plt.scatter(Tdf["c1"], Tdf["c2"], c="b")
plt.title("PCA scatter plot, Good")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.show()


#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



