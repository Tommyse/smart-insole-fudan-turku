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
import sklearn
from sklearn.decomposition import PCA
import scipy.stats as stats

from sklearn.cluster import KMeans
#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
from sklearn.metrics import silhouette_score
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

plt.style.use(['ggplot'])
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.13)
plt.gcf().subplots_adjust(left=0.13)
plt.rcParams["figure.figsize"] = (14,12)
plt.ticklabel_format(style='plain', useOffset=False)


#%%
data = pd.read_csv('../tommi_test_data.csv', sep=";", header=0)
#data = data.fillna(0)
data = data.dropna()
basedf = data

#removed errors
basedf = basedf.loc[basedf["Warning_code"] == 0]
data = data.loc[data["Warning_code"] == 0]
#data


#%%
#data = data.drop(labels=["App_time","Left/Right"], axis=1)
#data = data.astype('int')
#data["App_time"] = basedf["App_time"]
#data["Left/Right"] = basedf["Left/Right"]


#%%
#data1 = data.loc[data["session"] == 1]
data1 = data
data1L = data.loc[data["Left/Right"] == "L"]
data1R = data.loc[data["Left/Right"] == "R"]


#%%
plt.boxplot([data1L["S0_force"],data1L["S1_force"],data1L["S2_force"],data1L["S3_force"],data1L["S4_force"],data1L["S5_force"],data1L["S6_force"]])
plt.ylabel('force (N)')
plt.xlabel('sensors')
plt.title('Left foot')
plt.show()

plt.boxplot([data1R["S0_force"],data1R["S1_force"],data1R["S2_force"],data1R["S3_force"],data1R["S4_force"],data1R["S5_force"],data1R["S6_force"]])
plt.ylabel('force (N)')
plt.xlabel('sensors')
plt.title('Right foot')
plt.show()

plt.boxplot([data1["S0_force"],data1["S1_force"],data1["S2_force"],data1["S3_force"],data1["S4_force"],data1["S5_force"],data1["S6_force"]])
plt.ylabel('force (N)')
plt.xlabel('sensors')
plt.title('Both feet')
plt.show()


#%%
plt.boxplot([data1L["S0_start_time"],data1L["S1_start_time"],data1L["S2_start_time"],data1L["S3_start_time"],data1L["S4_start_time"],data1L["S5_start_time"],data1L["S6_start_time"]])
plt.ylabel('start_time (ms)')
plt.xlabel('sensors')
plt.title('Left foot')
plt.show()

plt.boxplot([data1R["S0_start_time"],data1R["S1_start_time"],data1R["S2_start_time"],data1R["S3_start_time"],data1R["S4_start_time"],data1R["S5_start_time"],data1R["S6_start_time"]])
plt.ylabel('start_time (ms)')
plt.xlabel('sensors')
plt.title('Right foot')
plt.show()


#%%
plt.boxplot([data1L["S0_max_time"],data1L["S1_max_time"],data1L["S2_max_time"],data1L["S3_max_time"],data1L["S4_max_time"],data1L["S5_max_time"],data1L["S6_max_time"]])
plt.ylabel('max_time (ms)')
plt.xlabel('sensors')
plt.title('Left foot')
plt.show()

plt.boxplot([data1R["S0_max_time"],data1R["S1_max_time"],data1R["S2_max_time"],data1R["S3_max_time"],data1R["S4_max_time"],data1R["S5_max_time"],data1R["S6_max_time"]])
plt.ylabel('max_time (ms)')
plt.xlabel('sensors')
plt.title('Right foot')
plt.show()


#%%
plt.boxplot([data1L["S0_end_time"],data1L["S1_end_time"],data1L["S2_end_time"],data1L["S3_end_time"],data1L["S4_end_time"],data1L["S5_end_time"],data1L["S6_end_time"]])
plt.ylabel('end_time (ms)')
plt.xlabel('sensors')
plt.title('Left foot')
plt.show()

plt.boxplot([data1R["S0_end_time"],data1R["S1_end_time"],data1R["S2_end_time"],data1R["S3_end_time"],data1R["S4_end_time"],data1R["S5_end_time"],data1R["S6_end_time"]])
plt.ylabel('end_time (ms)')
plt.xlabel('sensors')
plt.title('Right foot')
plt.show()


#%%
plt.boxplot([data1L["F1_force"],data1L["F2_force"],data1L["F3_force"]])
plt.ylabel('force (N)')
plt.xlabel('sensors')
plt.title('Left foot')
plt.show()

plt.boxplot([data1R["F1_force"],data1R["F2_force"],data1R["F3_force"]])
plt.ylabel('force (N)')
plt.xlabel('sensors')
plt.title('Right foot')
plt.show()


#%%
plt.boxplot([data1L["F1_time"],data1L["F2_time"],data1L["F3_time"]])
plt.ylabel('time')
plt.xlabel('sensors')
plt.title('Left foot')
plt.show()

plt.boxplot([data1R["F1_time"],data1R["F2_time"],data1R["F3_time"]])
plt.ylabel('time')
plt.xlabel('sensors')
plt.title('Right foot')
plt.show()


#%%
pd.DataFrame.describe(data1L)


#%%
pd.DataFrame.describe(data1R)


#%%
plt.hist(data1["Insole_timer"])
plt.xlabel('Insole_timer')
plt.ylabel('Count')
plt.show()

plt.hist(data1["Contact_time"])
plt.xlabel('Contact_time')
plt.ylabel('Count')
plt.show()

plt.hist(data1["Insole_timer"])
plt.xlabel('Insole_timer')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S0_force"])
plt.xlabel('S0_force')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S0_start_time"])
plt.xlabel('S0_start_time')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S0_max_time"])
plt.xlabel('S0_max_time')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S0_end_time"])
plt.xlabel('S0_end_time')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S1_force"])
plt.xlabel('S1_force')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S1_start_time"])
plt.xlabel('S1_start_time')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S1_max_time"])
plt.xlabel('S1_max_time')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S1_end_time"])
plt.xlabel('S1_end_time')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S2_force"])
plt.xlabel('S2_force')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S2_start_time"])
plt.xlabel('S2_start_time')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S2_max_time"])
plt.xlabel('S2_max_time')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S2_end_time"])
plt.xlabel('S2_end_time')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S3_force"])
plt.xlabel('S3_force')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S3_start_time"])
plt.xlabel('S3_start_time')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S3_max_time"])
plt.xlabel('S3_max_time')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S3_end_time"])
plt.xlabel('S3_end_time')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S4_force"])
plt.xlabel('S4_force')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S4_start_time"])
plt.xlabel('S4_start_time')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S4_max_time"])
plt.xlabel('S4_max_time')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S4_end_time"])
plt.xlabel('S4_end_time')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S5_force"])
plt.xlabel('S5_force')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S5_start_time"])
plt.xlabel('S5_start_time')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S5_max_time"])
plt.xlabel('S5_max_time')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S5_end_time"])
plt.xlabel('S5_end_time')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S6_force"])
plt.xlabel('S6_force')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S6_start_time"])
plt.xlabel('S6_start_time')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S6_max_time"])
plt.xlabel('S6_max_time')
plt.ylabel('Count')
plt.show()

plt.hist(data1["S6_end_time"])
plt.xlabel('S6_end_time')
plt.ylabel('Count')
plt.show()

plt.hist(data1["F1_force"])
plt.xlabel('F1_force')
plt.ylabel('Count')
plt.show()

plt.hist(data1["F1_time"])
plt.xlabel('F1_time')
plt.ylabel('Count')
plt.show()

plt.hist(data1["F2_force"])
plt.xlabel('F2_force')
plt.ylabel('Count')
plt.show()

plt.hist(data1["F2_time"])
plt.xlabel('F2_time')
plt.ylabel('Count')
plt.show()

plt.hist(data1["F3_force"])
plt.xlabel('F3_force')
plt.ylabel('Count')
plt.show()

plt.hist(data1["F3_time"])
plt.xlabel('F3_time')
plt.ylabel('Count')
plt.show()


#%%

#"Insole_timer"
values_cols = ["Contact_time","S0_force","S0_start_time","S0_max_time","S0_end_time","S1_force",
               "S1_start_time","S1_max_time","S2_force","S2_start_time","S2_max_time","S2_end_time",
               "S3_force","S3_start_time","S3_max_time","S3_end_time","S4_force","S4_start_time",
               "S4_max_time","S4_end_time","S5_force","S5_start_time","S5_max_time","S5_end_time",
               "S6_force","S6_start_time","S6_max_time","S6_end_time","F1_force","F1_time","F2_force",
               "F2_time","F3_force","F3_time"]

force_cols = ["S0_force","S1_force","S2_force","S3_force","S4_force","S5_force","S6_force"]
startT_cols = ["S0_start_time","S1_start_time","S2_start_time","S3_start_time","S4_start_time","S5_start_time","S6_start_time"]
maxT_cols = ["S0_max_time","S1_max_time","S2_max_time","S3_max_time","S4_max_time","S5_max_time","S6_max_time"]
endT_cols = ["S0_end_time","S1_end_time","S2_end_time","S3_end_time","S4_end_time","S5_end_time","S6_end_time"]


#%%
values1L = data1L
values1R = data1R


#%%
valuesL = data1L.loc[:,force_cols]
valuesR = data1R.loc[:,force_cols]

pca1=PCA(n_components=2, svd_solver='full')
pca1.fit(valuesL)

pca2=PCA(n_components=2, svd_solver='full')
pca2.fit(valuesR)

T1=pca1.transform(valuesL)
T2=pca2.transform(valuesR)

#Dataframe from T
T1df=pd.DataFrame(T1)
T1df.columns=["c1","c2"]
T2df=pd.DataFrame(T2)
T2df.columns=["c1","c2"]

plt.scatter(T1df["c1"], T1df["c2"], c="b")
plt.title("PCA scatter plot, left foot force")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.show()

plt.scatter(T2df["c1"], T2df["c2"], c="b")
plt.title("PCA scatter plot, right foot force")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.show()


#%%
best_k=0
best_s=0

#Picking best k
for k in range(2,11): #from 2 to 10
    kmeans=KMeans(n_clusters=k)
    clusters=kmeans.fit(valuesL)

    T=clusters.transform(valuesL)

    #Dataframe from T
    Tdf=pd.DataFrame(T)
    labels=clusters.labels_
    s_score=silhouette_score(Tdf, labels)
    print("k =",k,": ",round(s_score, 3))

    if(s_score>best_s):
        best_s=s_score
        best_k=k
    
print("\nBest k=",best_k)
print("Best silhouette score=",round(best_s, 3))

###Rerunning kmeans with best k
kmeans=KMeans(n_clusters=best_k)
clusters=kmeans.fit(valuesL)

T=clusters.transform(valuesL)

#Dataframe from T
Tdf=pd.DataFrame(T)
y_pred=kmeans.fit_predict(Tdf.values)

#print(Tdf2)
#print(labels)
#print(y_pred)

#PCA
pca=PCA(n_components=2, svd_solver='full')
pca.fit(Tdf)
T=pca.transform(Tdf)

#Dataframe from T
Tdf=pd.DataFrame(T)
Tdf.columns=["c1","c2"]

#Plot with prediction colors
plt.scatter(Tdf["c1"], Tdf["c2"], c=y_pred)
plt.title("Cluster scatter plot, left foot force")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.show()

best_k=0
best_s=0

#Picking best k
for k in range(2,11): #from 2 to 10
    kmeans=KMeans(n_clusters=k)
    clusters=kmeans.fit(valuesR)

    T=clusters.transform(valuesR)

    #Dataframe from T
    Tdf=pd.DataFrame(T)
    labels=clusters.labels_
    s_score=silhouette_score(Tdf, labels)
    print("k =",k,": ",round(s_score, 3))

    if(s_score>best_s):
        best_s=s_score
        best_k=k
    
print("\nBest k=",best_k)
print("Best silhouette score=",round(best_s, 3))

###Rerunning kmeans with best k
kmeans=KMeans(n_clusters=best_k)
clusters=kmeans.fit(valuesR)

T=clusters.transform(valuesR)

#Dataframe from T
Tdf=pd.DataFrame(T)
y_pred=kmeans.fit_predict(Tdf.values)

#print(Tdf2)
#print(labels)
#print(y_pred)

#PCA
pca=PCA(n_components=2, svd_solver='full')
pca.fit(Tdf)
T=pca.transform(Tdf)

#Dataframe from T
Tdf=pd.DataFrame(T)
Tdf.columns=["c1","c2"]

#Plot with prediction colors
plt.scatter(Tdf["c1"], Tdf["c2"], c=y_pred)
plt.title("Cluster scatter plot, right foot force")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.show()


#%%
valuesL = data1L.loc[:,startT_cols]
valuesR = data1R.loc[:,startT_cols]

pca1=PCA(n_components=2, svd_solver='full')
pca1.fit(valuesL)

pca2=PCA(n_components=2, svd_solver='full')
pca2.fit(valuesR)

T1=pca1.transform(valuesL)
T2=pca2.transform(valuesR)

#Dataframe from T
T1df=pd.DataFrame(T1)
T1df.columns=["c1","c2"]
T2df=pd.DataFrame(T2)
T2df.columns=["c1","c2"]

plt.scatter(T1df["c1"], T1df["c2"], c="b")
plt.title("PCA scatter plot, left foot start time")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.show()

plt.scatter(T2df["c1"], T2df["c2"], c="b")
plt.title("PCA scatter plot, right foot start time")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.show()


#%%
best_k=0
best_s=0

#Picking best k
for k in range(2,11): #from 2 to 10
    kmeans=KMeans(n_clusters=k)
    clusters=kmeans.fit(valuesL)

    T=clusters.transform(valuesL)

    #Dataframe from T
    Tdf=pd.DataFrame(T)
    labels=clusters.labels_
    s_score=silhouette_score(Tdf, labels)
    print("k =",k,": ",round(s_score, 3))

    if(s_score>best_s):
        best_s=s_score
        best_k=k
    
print("\nBest k=",best_k)
print("Best silhouette score=",round(best_s, 3))

###Rerunning kmeans with best k
kmeans=KMeans(n_clusters=best_k)
clusters=kmeans.fit(valuesL)

T=clusters.transform(valuesL)

#Dataframe from T
Tdf=pd.DataFrame(T)
y_pred=kmeans.fit_predict(Tdf.values)

#print(Tdf2)
#print(labels)
#print(y_pred)

#PCA
pca=PCA(n_components=2, svd_solver='full')
pca.fit(Tdf)
T=pca.transform(Tdf)

#Dataframe from T
Tdf=pd.DataFrame(T)
Tdf.columns=["c1","c2"]

#Plot with prediction colors
plt.scatter(Tdf["c1"], Tdf["c2"], c=y_pred)
plt.title("Cluster scatter plot, left foot start time")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.show()

best_k=0
best_s=0

#Picking best k
for k in range(2,11): #from 2 to 10
    kmeans=KMeans(n_clusters=k)
    clusters=kmeans.fit(valuesR)

    T=clusters.transform(valuesR)

    #Dataframe from T
    Tdf=pd.DataFrame(T)
    labels=clusters.labels_
    s_score=silhouette_score(Tdf, labels)
    print("k =",k,": ",round(s_score, 3))

    if(s_score>best_s):
        best_s=s_score
        best_k=k
    
print("\nBest k=",best_k)
print("Best silhouette score=",round(best_s, 3))

###Rerunning kmeans with best k
kmeans=KMeans(n_clusters=best_k)
clusters=kmeans.fit(valuesR)

T=clusters.transform(valuesR)

#Dataframe from T
Tdf=pd.DataFrame(T)
y_pred=kmeans.fit_predict(Tdf.values)

#print(Tdf2)
#print(labels)
#print(y_pred)

#PCA
pca=PCA(n_components=2, svd_solver='full')
pca.fit(Tdf)
T=pca.transform(Tdf)

#Dataframe from T
Tdf=pd.DataFrame(T)
Tdf.columns=["c1","c2"]

#Plot with prediction colors
plt.scatter(Tdf["c1"], Tdf["c2"], c=y_pred)
plt.title("Cluster scatter plot, right foot start time")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.show()


#%%
valuesL = data1L.loc[:,maxT_cols]
valuesR = data1R.loc[:,maxT_cols]

pca1=PCA(n_components=2, svd_solver='full')
pca1.fit(valuesL)

pca2=PCA(n_components=2, svd_solver='full')
pca2.fit(valuesR)

T1=pca1.transform(valuesL)
T2=pca2.transform(valuesR)

#Dataframe from T
T1df=pd.DataFrame(T1)
T1df.columns=["c1","c2"]
T2df=pd.DataFrame(T2)
T2df.columns=["c1","c2"]

plt.scatter(T1df["c1"], T1df["c2"], c="b")
plt.title("PCA scatter plot, left foot max time")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.show()

plt.scatter(T2df["c1"], T2df["c2"], c="b")
plt.title("PCA scatter plot, right foot max time")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.show()


#%%
best_k=0
best_s=0

#Picking best k
for k in range(2,11): #from 2 to 10
    kmeans=KMeans(n_clusters=k)
    clusters=kmeans.fit(valuesL)

    T=clusters.transform(valuesL)

    #Dataframe from T
    Tdf=pd.DataFrame(T)
    labels=clusters.labels_
    s_score=silhouette_score(Tdf, labels)
    print("k =",k,": ",round(s_score, 3))

    if(s_score>best_s):
        best_s=s_score
        best_k=k
    
print("\nBest k=",best_k)
print("Best silhouette score=",round(best_s, 3))

###Rerunning kmeans with best k
kmeans=KMeans(n_clusters=best_k)
clusters=kmeans.fit(valuesL)

T=clusters.transform(valuesL)

#Dataframe from T
Tdf=pd.DataFrame(T)
y_pred=kmeans.fit_predict(Tdf.values)

#print(Tdf2)
#print(labels)
#print(y_pred)

#PCA
pca=PCA(n_components=2, svd_solver='full')
pca.fit(Tdf)
T=pca.transform(Tdf)

#Dataframe from T
Tdf=pd.DataFrame(T)
Tdf.columns=["c1","c2"]

#Plot with prediction colors
plt.scatter(Tdf["c1"], Tdf["c2"], c=y_pred)
plt.title("Cluster scatter plot, left foot max time")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.show()

best_k=0
best_s=0

#Picking best k
for k in range(2,11): #from 2 to 10
    kmeans=KMeans(n_clusters=k)
    clusters=kmeans.fit(valuesR)

    T=clusters.transform(valuesR)

    #Dataframe from T
    Tdf=pd.DataFrame(T)
    labels=clusters.labels_
    s_score=silhouette_score(Tdf, labels)
    print("k =",k,": ",round(s_score, 3))

    if(s_score>best_s):
        best_s=s_score
        best_k=k
    
print("\nBest k=",best_k)
print("Best silhouette score=",round(best_s, 3))

###Rerunning kmeans with best k
kmeans=KMeans(n_clusters=best_k)
clusters=kmeans.fit(valuesR)

T=clusters.transform(valuesR)

#Dataframe from T
Tdf=pd.DataFrame(T)
y_pred=kmeans.fit_predict(Tdf.values)

#print(Tdf2)
#print(labels)
#print(y_pred)

#PCA
pca=PCA(n_components=2, svd_solver='full')
pca.fit(Tdf)
T=pca.transform(Tdf)

#Dataframe from T
Tdf=pd.DataFrame(T)
Tdf.columns=["c1","c2"]

#Plot with prediction colors
plt.scatter(Tdf["c1"], Tdf["c2"], c=y_pred)
plt.title("Cluster scatter plot, right foot max time")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.show()


#%%
valuesL = data1L.loc[:,endT_cols]
valuesR = data1R.loc[:,endT_cols]

pca1=PCA(n_components=2, svd_solver='full')
pca1.fit(valuesL)

pca2=PCA(n_components=2, svd_solver='full')
pca2.fit(valuesR)

T1=pca1.transform(valuesL)
T2=pca2.transform(valuesR)

#Dataframe from T
T1df=pd.DataFrame(T1)
T1df.columns=["c1","c2"]
T2df=pd.DataFrame(T2)
T2df.columns=["c1","c2"]

plt.scatter(T1df["c1"], T1df["c2"], c="b")
plt.title("PCA scatter plot, left foot end time")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.show()

plt.scatter(T2df["c1"], T2df["c2"], c="b")
plt.title("PCA scatter plot, right foot end time")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.show()


#%%
best_k=0
best_s=0

#Picking best k
for k in range(2,11): #from 2 to 10
    kmeans=KMeans(n_clusters=k)
    clusters=kmeans.fit(valuesL)

    T=clusters.transform(valuesL)

    #Dataframe from T
    Tdf=pd.DataFrame(T)
    labels=clusters.labels_
    s_score=silhouette_score(Tdf, labels)
    print("k =",k,": ",round(s_score, 3))

    if(s_score>best_s):
        best_s=s_score
        best_k=k
    
print("\nBest k=",best_k)
print("Best silhouette score=",round(best_s, 3))

###Rerunning kmeans with best k
kmeans=KMeans(n_clusters=best_k)
clusters=kmeans.fit(valuesL)

T=clusters.transform(valuesL)

#Dataframe from T
Tdf=pd.DataFrame(T)
y_pred=kmeans.fit_predict(Tdf.values)

#print(Tdf2)
#print(labels)
#print(y_pred)

#PCA
pca=PCA(n_components=2, svd_solver='full')
pca.fit(Tdf)
T=pca.transform(Tdf)

#Dataframe from T
Tdf=pd.DataFrame(T)
Tdf.columns=["c1","c2"]

#Plot with prediction colors
plt.scatter(Tdf["c1"], Tdf["c2"], c=y_pred)
plt.title("Cluster scatter plot, left foot end time")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.show()

best_k=0
best_s=0

#Picking best k
for k in range(2,11): #from 2 to 10
    kmeans=KMeans(n_clusters=k)
    clusters=kmeans.fit(valuesR)

    T=clusters.transform(valuesR)

    #Dataframe from T
    Tdf=pd.DataFrame(T)
    labels=clusters.labels_
    s_score=silhouette_score(Tdf, labels)
    print("k =",k,": ",round(s_score, 3))

    if(s_score>best_s):
        best_s=s_score
        best_k=k
    
print("\nBest k=",best_k)
print("Best silhouette score=",round(best_s, 3))

###Rerunning kmeans with best k
kmeans=KMeans(n_clusters=best_k)
clusters=kmeans.fit(valuesR)

T=clusters.transform(valuesR)

#Dataframe from T
Tdf=pd.DataFrame(T)
y_pred=kmeans.fit_predict(Tdf.values)

#print(Tdf2)
#print(labels)
#print(y_pred)

#PCA
pca=PCA(n_components=2, svd_solver='full')
pca.fit(Tdf)
T=pca.transform(Tdf)

#Dataframe from T
Tdf=pd.DataFrame(T)
Tdf.columns=["c1","c2"]

#Plot with prediction colors
plt.scatter(Tdf["c1"], Tdf["c2"], c=y_pred)
plt.title("Cluster scatter plot, right foot end time")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.show()

