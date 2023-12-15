#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from  sklearn.model_selection import train_test_split
from sklearn import svm 
from sklearn.metrics import accuracy_score


# In[2]:


data=pd.read_csv("loan.csv")


# In[3]:


data


# In[4]:


data.shape


# In[5]:


data.head()


# In[6]:


data.describe()


# In[7]:


data.isnull().sum()


# In[8]:


data = data.dropna()


# In[9]:


data.isnull().sum()


# In[10]:


data.replace({'Loan_Status':{'N':0,'Y':1}},inplace=True)


# In[11]:


data.head()


# In[12]:


data['Dependents'].value_counts()


# In[13]:


data=data.replace(to_replace='3+',value=4)


# In[14]:


data['Dependents'].value_counts()


# In[15]:


sns.countplot(x='Education',hue='Loan_Status',data=data)


# In[16]:


sns.countplot(x='Married',hue='Loan_Status',data=data)


# In[17]:


sns.countplot(x='Gender',hue='Loan_Status',data=data)


# In[18]:


data.head(2)


# In[19]:


sns.countplot(x='Self_Employed',hue='Loan_Status',data=data)


# In[20]:


sns.countplot(x='Property_Area',hue='Loan_Status',data=data)


# In[21]:


data.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Education':{'Graduate':1,'Not Graduate':0},'Self_Employed':{'No':0,'Yes':1}
             ,'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2}},inplace=True)


# In[22]:


data.head(3)


# In[23]:


X=data.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y=data['Loan_Status']


# In[24]:


print(X)
print(Y)


# In[25]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=2)


# In[26]:


print(X.shape,X_train.shape,X_test.shape)


# In[27]:


X_train.shape


# In[28]:


X_test.shape


# In[29]:


Y_train.shape


# In[30]:


Y_test.shape

Model
# In[31]:


classifier=svm.SVC(kernel='linear')


# In[32]:


classifier.fit(X_train,Y_train)


# In[33]:


prd=classifier.predict(X_train)


# In[34]:


accuracy_score=accuracy_score(prd,Y_train)


# In[35]:


print('Accuracy_score:',accuracy_score)


# In[ ]:




