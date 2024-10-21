#%% md
# # Breast Cancer Classification Using Neural Network
#%%
# Import Necessary Libraries
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
#%% md
# ## Load Dataset
#%%

  
# fetch dataset 
breast_cancer = fetch_ucirepo(id=14) 
  
# data (as pandas dataframes) 
X = breast_cancer.data.features 
y = breast_cancer.data.targets 
  
# metadata 
print(breast_cancer.metadata) 
  
# variable information 
print(breast_cancer.variables) 

#%%

Merge = pd.concat([X, y], axis=1)
Merge.to_csv('breast_cancer.csv', index=False)
#%%
data = pd.read_csv('breast_cancer.csv')
#%%
data.head()
#%%
data.describe()
#%%
data.shape
#%%
data.info()
#%%
data.isnull().sum()
#%%
data.dtypes
#%%
# Filling missing values in 'node-caps' and 'breast-quad' with the mode (most frequent value)
data['node-caps'].fillna(data['node-caps'].mode()[0], inplace=True)
data['breast-quad'].fillna(data['breast-quad'].mode()[0], inplace=True)

#%%
data.isnull().sum()
#%%
data.head()