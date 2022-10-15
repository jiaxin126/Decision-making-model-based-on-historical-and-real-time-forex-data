#!/usr/bin/env python
# coding: utf-8

# In[1]:


Jiaxin Wang


# In[ ]:


#libraries
import requests
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from datetime import datetime
import sqlite3
import sqlalchemy
import seaborn as sns
import time
import matplotlib.pyplot as plt
from arctic import Arctic
import pywt
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


# # Historical Data

# In[2]:


#the data 
#read data from sqlite 
conn = sqlite3.connect('/Users/wangjiaxin/databasename.db')
result_df = pd.read_sql_query("select * from currency_exchange_rates;", conn)
conn.close()
# Connect to Local MONGODB
store = Arctic('localhost')
# create the library 
store.initialize_library('project')
# access the library
library = store['project']
library.write('project',result_df)
#read data
df_all=library.read('project').data
df_all['ts']=pd.to_datetime(df_all['timestamp']).apply(lambda a: a.timestamp())
df_all.head()


# ### USD/AUD

# In[3]:


df_USDAUD=df_all[df_all['currency_pair']=='USD/AUD'][['fx_rate','ts']]
#check if there are missing values
df_USDAUD.isnull().sum()


# ### create signal by calculating it's return

# In[4]:


#percentage change formula, (new value − old value)/old value
return_list=list()
for i in range(1,len(df_USDAUD)) :
    return_list.append((df_USDAUD.fx_rate.iloc[i] - df_USDAUD.fx_rate.iloc[i-1])/df_USDAUD.fx_rate.iloc[i-1])
return_list.insert(0,df_USDAUD.fx_rate.iloc[0])
df_USDAUD["return"]=return_list
df_USDAUD


# cwt:wavelet decomposition
# non-stationary signals(time series) are very complex and prone to noise and misleading values.
# improve the prediction accuracy of a model

# In[5]:


signal = df_USDAUD["return"].iloc[1:].reset_index(drop=True)
wavelet_name = "gaus5"
scales = 2 ** np.arange(8) # taking scales from  1 to 128,i.e,  [  1,   2,   4,   8,  16,  32,  64, 128]
#using the continuous wavelet function to transform the signal
coef, freq = pywt.cwt(signal, scales, wavelet_name)
df_coef = pd.DataFrame(coef).T
df_coef.columns = [str(int(i)) for i in 1/freq]


for j in df_coef.columns:
    for i in range(1,10):
        df_coef[j + "_lag_" + str(i)] = df_coef[j].shift(i)

df_coef["return"]=df_USDAUD["return"]
df_coef = df_coef.dropna(axis=0)
df_coef


# In[6]:


plt.plot(df_coef['128'])


# ### Spectral-Clustering

# In[7]:


from sklearn.cluster import SpectralClustering
x=df_coef.iloc[:,:80]
params = {"n_clusters": 4}
spectral = SpectralClustering(n_clusters = params['n_clusters'],
                                      eigen_solver = "arpack",
                                      affinity = "nearest_neighbors",
                                      assign_labels = "discretize",
                                      random_state = 42)

#preprocess data
from sklearn.preprocessing import StandardScaler
X = np.array(x)
X = StandardScaler().fit_transform(X)


spectral.fit(X)
y_pred = spectral.labels_.astype(np.int)
df_coef['regime'] = y_pred

df_coef.reset_index(inplace=True)
df_coef


# In[8]:


df_coef.regime.value_counts()


# ### Decision Stratery

# In[9]:


a0=pd.DataFrame()
a0["cluster number"]=[0,1,2,3]
a0["Average"]=[np.mean(df_coef[df_coef.regime==0]["return"]),np.mean(df_coef[df_coef.regime==1]["return"]),
                 np.mean(df_coef[df_coef.regime==2]["return"]),np.mean(df_coef[df_coef.regime==3]["return"])]
a0["Standard_Deviation"]=[np.std(df_coef[df_coef.regime==0]["return"]),np.std(df_coef[df_coef.regime==1]["return"]),
                 np.std(df_coef[df_coef.regime==2]["return"]),np.std(df_coef[df_coef.regime==3]["return"])]
a0["Currency"]=["EUR-GBP","EUR-GBP","EUR-GBP","EUR-GBP"]

a0


# ### preprocessing the data to make it visualizable

# In[10]:


# Normalizing the Data
X_normalized = normalize(X)

# Converting the numpy array into a pandas DataFrame
X_normalized = pd.DataFrame(X_normalized)


# In[11]:


# Reducing the dimensions of the data
pca = PCA(n_components = 2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']

X_principal.head()


# Building the clustering model
spectral_model_knn = SpectralClustering(n_clusters = 4, affinity ="nearest_neighbors")
  
# Training the model and Storing the predicted cluster labels
labels_knn = spectral_model_knn.fit_predict(X_principal)

# Building the label to colour mapping
colours = {}
colours[0] = 'b'
colours[1] = 'y'
colours[2] = 'r'
colours[3] = 'g'

# Building the colour vector for each data point
cvec = [colours[label] for label in labels_knn]

plt.figure(figsize =(9, 9))
plt.scatter(X_principal['P1'], X_principal['P2'], c = cvec)
plt.show()


# # build and compare classifier 

# In[12]:


import sklearn
X=np.array(df_coef.iloc[:,81])
Y=np.array(df_coef.iloc[:,82])


# ## Knearest Neighbour

# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)


# In[14]:


#coloumn
X_train=X_train.reshape(-1,1)
y_train=y_train.reshape(-1,1)
X_test=X_test.reshape(-1,1)


# In[15]:


#propressing data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[16]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 13, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


# In[17]:


y_pred = classifier.predict(X_test)


# In[18]:


#the accuracy of this classifier
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac_knn = accuracy_score(y_test,y_pred)
print("ACCURACY OF KNN MODEL: ", ac_knn)


# ## Random Forest

# In[19]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 100) 
 
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)
 
# performing predictions on the test dataset
y_pred = clf.predict(X_test)
 
# metrics are used to find accuracy or error
from sklearn import metrics 
print()
 
# using metrics module for accuracy calculation
print("ACCURACY OF RANDOM FOREST MODEL: ", metrics.accuracy_score(y_test, y_pred))


# ## Decision Tree

# In[20]:


from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("ACCURACY OF DECISION TREE MODEL: ", metrics.accuracy_score(y_test, y_pred))


# # Real-time Data

# In[21]:


rt_USDAUD=pd.DataFrame(columns=["FX_Rate","Timestamp"])
i=0
j=1639394133000000000
k=0
while k < 500 :
    api_url_e="https://api.polygon.io/vX/quotes/C:USDAUD?timestamp={}&apiKey=beBybSi8daPgsTp5yx5cHtHpYcrjp5Jq"
    url = api_url_e.format(j)
    data_USDAUD= requests.get(url).json()
    try :
        a=data_USDAUD["results"][0]["ask_price"]
        b=data_USDAUD["results"][0]["participant_timestamp"]
        rt_USDAUD.loc[i] = np.array([a, b])
        
    
    except :
        a=np.nan
        b=j
        rt_USDAUD.loc[i]=np.array([a,b])
    
    i+=1
    j+=360000000000
    k+=1
    
rt_USDAUD


# ### Saving Data

# In[22]:


# Creating a connection to the local port
store = Arctic('localhost')

#Initializing library
store.initialize_library('project')

#Accessing the library
library = store['project']

#Creating a symbol and storing data
library.write('USDAUD_Real_time',rt_USDAUD, metadata={'source': 'polygon'})


# ### Sanity Check

# In[23]:


# Let's use some more sanity checks to understand more about the data: 
# get amount of rows and columns
print(rt_USDAUD.shape)
# get columns in the dataframe
print(rt_USDAUD.columns)

# Now, use the .describe() method to easily get some summary statistics:
# get statistical summary
rt_USDAUD.describe()
# view in transposed form
print(rt_USDAUD.describe().transpose())

# Let's use .min(), .max(), .mean(), and .median() methods as well:
# get max and min values
rt_USDAUD.max()
rt_USDAUD.min()
rt_USDAUD.mean()
print(rt_USDAUD.median())


# In[24]:


# Histograms
sns.distplot(rt_USDAUD['FX_Rate'])


# In[25]:


#missing value
rt_USDAUD.isnull().sum()


# In[26]:


#forward fill
rt_USDAUD.ffill(inplace=True)


# In[27]:


rt_USDAUD.isnull().sum()


# ### Scaling

# In[28]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# standardize the data and store in out_scaled numpy array
reshaped_fx_rate=np.array(rt_USDAUD.FX_Rate).reshape(-1,1)
out_scaled_ = scaler.fit_transform(reshaped_fx_rate)
print(out_scaled_)


# ### Testing on real data

# ### KNN

# In[29]:


final_output_knn=classifier.predict(out_scaled_)
final_output_knn


# ### RFE

# In[30]:


final_output_rfe=clf.predict(out_scaled_)
final_output_rfe


# ### DT

# In[31]:


final_output_dt=clf.predict(out_scaled_)
final_output_dt


# ### Buy-Sell-Do Nothing

# In[32]:


decision_list=list()
buy_in=int(a0[a0.Standard_Deviation==min(a0[a0.Average>0].Standard_Deviation)]["cluster number"])
sell=int(a0[a0.Standard_Deviation==min(a0[a0.Average<0].Standard_Deviation)]["cluster number"])


for i in final_output_rfe:
    if i==buy_in:
        decision_list.append("buy in")
        
    elif i==sell :
        decision_list.append("sell")
        
    else:
        decision_list.append("do nothing")

        
print(decision_list)


# In[34]:


sell=0
buy=0
nothing=0
for i in decision_list:
    if i=="sell":
        sell+=1
    elif i=="buy in":
        buy+=1
    elif i=='do nothing':
        nothing+=1
    else:
        continue

print("No. of buys before the budget is exhausted : ",buy)
print("No. of sell before the budget is exhausted : ",sell)
print("No. of do nothing before the budget is exhausted : ",nothing)


# In[ ]:




