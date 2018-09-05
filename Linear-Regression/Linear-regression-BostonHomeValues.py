
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from sklearn.datasets import load_boston

boston = load_boston()
boston.keys()


# In[7]:


print(boston.DESCR)


# In[12]:


boston['feature_names']


# In[76]:


dataset = pd.DataFrame(boston.data, columns=boston['feature_names'])
dataset.drop(labels=['INDUS', 'AGE'], axis=1, inplace=True)
dataset.head()


# In[77]:


dataset["Home Value $1000's"] = boston['target']
dataset.head()


# In[27]:


dataset.shape


# In[91]:


sns.pairplot(dataset)


# In[89]:


dataset.corr('pearson')


# In[93]:


sns.heatmap(dataset.corr('pearson'))


# In[80]:


# train test split

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


# In[81]:


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[82]:


y_pred = regressor.predict(X_test)


# In[83]:


regressor.score(X_test, y_test)


# In[84]:


from sklearn.metrics import accuracy_score

regressor.score(X_test, y_test)


# In[85]:


pd.DataFrame(regressor.coef_, index=dataset.columns[:-1], columns=['Coefficient'])


# In[86]:


from sklearn import metrics

print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Square Error: ', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Square Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[12]:


# To calculate p-value
import statsmodels.api as sm
from scipy import stats


# In[87]:


X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

