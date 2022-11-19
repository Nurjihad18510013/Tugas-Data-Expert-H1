#!/usr/bin/env python
# coding: utf-8

# In[1]:


2 + 3 + 9


# In[2]:


99 - 73


# In[3]:


23.54 * -1432


# In[4]:


100 / 7


# In[5]:


100 // 7


# In[6]:


100 % 7


# In[7]:


5 ** 3


# In[9]:


((2 + 5 ) * (17 - 3))/(4**3)


# In[10]:


import numpy as np 
from sklearn import preprocessing

sample_data = np.array([[2.1, 1.9, 5.5],
                        [-1.5, 2.4, 3.5],
                        [0.5, -7.9, 5.6],
                        [5.9, 2.3, -5.8],])
sample_data


# In[11]:


sample_data.shape


# In[12]:


sample_data


# In[14]:


preprocessor = preprocessing.Binarizer(threshold=0.5)
binarised_data = preprocessor.transform(sample_data)
binarised_data


# In[16]:


label_kategori = ['senin', 'selasa', 'rabu', 'kamis', 'jumat', 'sabtu', 'minggu']
encoder = preprocessing.LabelEncoder()
encoder.fit(label_kategori)

print("\nLabel mapping")
for i, item in enumerate(encoder.classes_):
    print(item, '=', i)


# In[21]:


from sklearn.model_selection import train_test_split

X_data = range(10)
y_data = range(10)

print("random_state ditentukan")
for i in range(3):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3, random_state = 42)
    print(y_test)
    


# In[22]:


print("random_state tidak ditentukan")
for i in range(3):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3, random_state = None)
    
print(y_test)


# # Latihan Sckit Learn

# In[23]:


import sklearn
from sklearn import datasets


# In[24]:


iris = datasets.load_iris()
iris


# In[25]:


x=iris.data
y=iris.target


# In[41]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[42]:


len(X_test)


# In[43]:


from sklearn import tree


# In[44]:


clf = tree.DecisionTreeClassifier()


# In[45]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, x, y, cv=2)


# In[46]:


scores


# In[47]:


from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

print(f'Dimensi Feature: {X.shape}')
print(f'Class: {set(y)}')


# In[48]:


load_iris


# In[49]:


data = load_iris()
data


# In[50]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[55]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=2)
model.fit(X_train, y_train)


# In[56]:


import matplotlib.pyplot as plt 
from sklearn import tree

plt.subplots(figsize=(10,10))
tree.plot_tree(model, fontsize=10)
plt.show()


# In[57]:


from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# # Latihan Linier Regression

# In[58]:


# import Library

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[64]:


# Membuat data Dummy
# data jumlah kamar

bedrooms = np.array([1,1,2,2,3,4,4,5,5,5])
house_price = np.array ([15000, 21000, 10000, 25000, 18000, 90000, 30000, 22000, 10200, 27000])


# In[65]:


# Menampilkan Scatter plot

plt.scatter(bedrooms, house_price)


# In[69]:


# import library
from sklearn.linear_model import LinearRegression

#melatih Model
bedrooms = bedrooms.reshape(-1,1)
linreg = LinearRegression()
linreg.fit(bedrooms, house_price)


# In[70]:


plt.scatter(bedrooms, house_price)
plt.plot(bedrooms, linreg.predict(bedrooms))


# # Logistic Regression

# In[71]:


import pandas as pd


# In[72]:


df = pd.read_csv('C:\Data_Expert\Social_Network_Ads.csv')
df.head()


# In[73]:


df.info()


# In[74]:


data = df.drop(columns=['User ID'])


# In[75]:


data = pd.get_dummies(data)
data


# In[76]:


predictions = ['Age', 'EstimatedSalary', 'Gender_Female', 'Gender_Male']

X = data[predictions]
y = data['Purchased']


# In[79]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
scaled_data = scaler.transform(X)
scaled_data = pd.DataFrame(scaled_data, columns= X.columns)
scaled_data


# In[80]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[81]:


from sklearn import linear_model
model = linear_model .LogisticRegression()
model.fit(X_train, y_train)


# In[82]:


model.score(X_test, y_test)


# # Latihan K-Means Clustering

# In[83]:


import pandas as pd


# In[84]:


df = pd.read_csv('C:\Data_Expert\Mall_Customers.csv')
df.head(3)


# In[86]:


df = df.rename(columns={'Gender': 'gender', 'Age': 'age', 'Annual Income (k$)': 'annual_income', 'Spending Score (1-100)': 'spending score'})

df['gender'].replace(['Female','Male'], [0,1], inplace=True)
df.head()


# In[87]:


X = df.drop(columns=['CustomerID', 'gender'], axis=1)


# In[90]:


from sklearn.cluster import KMeans


# In[91]:


clusters = []
for i in range(1,11):
    km = KMeans(n_clusters=i).fit(X)
    clusters.append(km.inertia_)


# In[98]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[99]:


fig, ax = plt.subplots(figsize=(8,4))
sns.lineplot(x=list(range(1,11)), y=clusters, ax=ax)
ax.set_title('Cari Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')


# In[100]:


km5 = KMeans(n_clusters=5).fit(X)

X['Labels'] = km5.labels_


# In[103]:


plt.figure(figsize=(8,4))
sns.scatterplot(X['annual_income'], X['spending score'], hue=X['Labels'],
               palette=sns.color_palette('hls', 5))

plt.title('KMeans dengan 5 Cluster')
plt.show()


# In[ ]:




