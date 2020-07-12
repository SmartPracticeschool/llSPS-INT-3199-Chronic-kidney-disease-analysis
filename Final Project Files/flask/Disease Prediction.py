#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[10]:


dataset = pd.read_csv("ckd.csv",header=0, na_values="?")


# In[11]:


dataset


# In[12]:


dataset.replace("?", np.NaN)


# In[13]:


cleanup = {"Rbc":     {"normal": 1, "abnormal": 0},
           "Pc": {"normal": 1, "abnormal": 0},
           "Pcc": {"present": 1, "notpresent": 0},
           "Ba": {"present": 1, "notpresent": 0},
           "Htn": {"yes": 1, "no": 0},
           "Dm": {"yes": 1, "no": 0},
           "Cad": {"yes": 1, "no": 0},
           "Appet": {"good": 1, "poor": 0},
           "pe": {"yes": 1, "no": 0},
           "Ane": {"yes": 1, "no": 0}}


# In[14]:


dataset.replace(cleanup, inplace=True)


# In[15]:


dataset.fillna(round(dataset.mean(),2), inplace=True)


# In[16]:


dataset


# In[66]:


x=dataset.iloc[:,1:11]


# In[67]:


y=dataset.iloc[:,-1]


# In[68]:


from sklearn.preprocessing import LabelEncoder


# In[69]:


lb=LabelEncoder()
y=lb.fit_transform(y)


# In[86]:


x


# In[87]:


feat_labels = ['Bp','Sg','Al','Su','Rbc','Pc','Pcc','Ba','Bgr','Bu']


# In[88]:


from sklearn.model_selection import train_test_split                #previously cros_validation was used in sklearn
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[85]:


x_train


# In[74]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[75]:


from joblib import dump
dump(sc,"scaling.save")


# In[76]:


from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier(criterion='entropy',random_state=0)


# In[77]:


dt.fit(x_train,y_train)


# In[78]:


import pickle
pickle.dump(dt,open('Prediction.pkl','wb'))


# In[79]:


from sklearn.feature_selection import SelectFromModel
for feature in zip(feat_labels, dt.feature_importances_):
    print(feature)


# In[80]:


y_predict=dt.predict(x_test)
y_predict


# In[81]:


y_test


# In[82]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:




