#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import sklearn
from matplotlib import pyplot as plt
import seaborn as sns 


# In[90]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error ,r2_score , mean_absolute_error


# In[3]:


from scipy import stats 
from scipy.stats import zscore


# In[4]:


data=pd.read_csv("tem.csv")


# In[5]:


data


# In[6]:


data.describe()


# In[7]:


data.info()


# In[8]:


data.isnull().sum()


# In[9]:


from sklearn.preprocessing import StandardScaler


# In[10]:


data['station']=data['station'].fillna(1.0)


# In[11]:


data.isnull().sum()


# In[12]:


data['Date']=data['Date'].fillna(13-8-2017)


# In[13]:


data.drop(columns=['lat','lon'],inplace=True)


# In[14]:


for col in data.columns:
    if col not in ['station','Date']:
        data[col]=data[col].replace(np.nan,data[col].mean())


# In[15]:


data['Date']=pd.to_datetime(data['Date'])


# In[16]:


data


# In[17]:


data_corr=data.corr()
plt.figure(figsize=(25,15))
sns.heatmap(data_corr,vmin=-1,vmax=1,annot=True,square=True,center=0,fmt='.2g',linewidths=0.1)
plt.tight_layout()


# In[18]:


data.skew()


# In[19]:


z_score=zscore(data[['Present_Tmax','Present_Tmin','LDAPS_RHmax','LDAPS_Tmax_lapse','LDAPS_Tmin_lapse','LDAPS_LH','DEM','Slope']])
abs_z_score=np.abs(z_score)
filtering_entry=(abs_z_score < 3).all(axis=1)
data=data[filtering_entry]
data.reset_index(inplace=True)


# In[20]:


data.skew()


# In[60]:


import sklearn
from sklearn.preprocessing import LabelEncoder
l_e=LabelEncoder()


# In[62]:


data['Date']=l_e.fit_transform(data['Date'])


# In[63]:


x=data.drop(['Next_Tmax','Next_Tmin'],axis=1)
y=data[['Next_Tmax','Next_Tmin']]


# In[64]:


for index in x.skew().index:
    if x.skew().loc[index]> 0.5:
        x[index]=np.cbrt(x[index])
        if x.skew().loc[index]<-0.5:
            x[index]=np.square(x[index])


# In[65]:


x.skew()


# In[66]:


scaler=StandardScaler()


# In[67]:


x


# In[68]:


y


# In[69]:


x_train,x_test,y_tarin,y_test=train_test_split(x,y,test_size=0.25,random_state=2)


# In[70]:


x_train.shape


# In[71]:


x_test.shape


# In[72]:


y_test.shape


# In[73]:


y_tarin.shape


# In[74]:


x_train


# In[75]:


y_tarin


# In[76]:


x_train['Date'].value_counts()


# In[77]:


x_train,x_test,y_tarin,y_test


# In[78]:


from sklearn.linear_model import LinearRegression


# In[79]:


reg= LinearRegression()


# In[80]:


reg.fit(x_train,y_tarin)


# In[85]:


pred=reg.predict(x_test)
pred.shape


# In[86]:


y_test.shape


# In[87]:


reg.score(x_train,y_tarin)


# In[104]:


reg.coef_


# In[91]:


print('error:')
print('Mean absolute error:',mean_absolute_error(y_test,pred))
print('Mean squared erroe:',mean_squared_error(y_test,pred))
print('Root mean squared Error:',np.sqrt(mean_squared_error(y_test,pred)))


# In[89]:


print(r2_score(y_test,pred))


# In[ ]:




