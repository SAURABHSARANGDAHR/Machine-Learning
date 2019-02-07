#!/usr/bin/env python
# coding: utf-8

# # Machine Learning

# In[38]:


from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
iris


# In[2]:


x= iris.data[:,[2,3]]    # taking only no. of petals and width of petals that is colum 2 & 3 of the entire dataset
x


# In[3]:


len(x)


# In[4]:


x.shape


# In[5]:


y = iris.target
y


# In[6]:


y.shape


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=1, stratify=y)


# In[9]:


x.shape


# In[10]:


X_train.shape


# In[11]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[12]:


from sklearn.preprocessing import StandardScaler


# In[13]:


X_test


# In[14]:


sc=StandardScaler()


# In[15]:


sc.fit(X_train)              #fit all the paramater or the data set to train


# In[16]:


X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# In[17]:


sc.transform([[0,0],[0,0]])


# In[18]:


X_train_std


# # Perceptron

# In[19]:


from sklearn.linear_model import Perceptron


# In[105]:


ppn = Perceptron(n_iter=4000,eta0=0.1, random_state=1)    #no. of iterations=4000; more alteration better resuult; eta is kept small


# In[106]:


ppn.fit(X_train_std,y_train)


# # Lets predict output of 'TEST' data

# In[107]:


y_pred = ppn.predict(X_test_std)


# In[108]:


(y_test != y_pred).sum()    # 3 elements do not match the output


# In[109]:


x_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))


# In[110]:


import vis
get_ipython().run_line_magic('matplotlib', 'inline')


# In[111]:


vis.plot_decision_regions(X=x_combined_std, y=y_combined, classifier = ppn, test_idx=range(105,150))
# in output , 3 types of flowers are shown, 3 or 4 lie near red & blue region,
# which is error that is linear model is not the correct model to classify the flowers,
#same case for green-blue partiion


# # Logistic Regression on same Data

# In[112]:


from sklearn.linear_model import LogisticRegression


# In[113]:


lr = LogisticRegression(C = 100.0, random_state=1)


# In[114]:


lr.fit(X_train_std, y_train)


# ### Predict

# In[115]:


y_pred= lr.predict(X_test_std)


# In[116]:


(y_test != y_pred).sum()


# In[117]:


vis.plot_decision_regions(X=x_combined_std, y=y_combined, classifier = lr, test_idx=range(105,150))


#  # SVM (Support Vector Method)

# In[118]:


from sklearn.svm import SVC


# In[119]:


svm = SVC(kernel= 'linear',C=1.0, random_state=1)


# In[120]:


svm.fit(X_train_std,y_train)


# # Predict on test data - SVM

# In[121]:


y_pred = svm.predict(X_test_std)


# In[122]:


(y_test != y_pred).sum()


# In[123]:


vis.plot_decision_regions(X=x_combined_std, y=y_combined, classifier = svm, test_idx=range(105,150))


# # Decicion Tree Learning on same adata

# In[124]:


from sklearn.tree import DecisionTreeClassifier


# In[125]:


tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)


# In[126]:


tree.fit(X_train_std, y_train)


# # Predict on test data - Decision tree on same data

# In[127]:


y_pred=tree.predict(X_test_std)


# In[128]:


(y_test != y_pred).sum()


# In[129]:


vis.plot_decision_regions(X=x_combined_std, y=y_combined, classifier = tree, test_idx=range(105,150))vis.plot_decision_regions(X=x_combined_std, y=y_combined, classifier = tree, test_idx=range(105,150))


# # Random Forest Learning on the same data 

# In[130]:


from sklearn.ensemble import RandomForestClassifier


# In[131]:


forest = RandomForestClassifier(criterion='gini', n_estimators=25, random_state=1)


# In[132]:


forest.fit(X_train_std, y_train)


# # Predict random forest on the same data

# In[133]:


y_pred=forest.predict(X_test_std)


# In[134]:


(y_test != y_pred).sum()


# In[135]:


vis.plot_decision_regions(X=x_combined_std, y=y_combined, classifier = forest, test_idx=range(105,150))


# In[143]:


from sklearn.neighbors import KNeighborsClassifier


# In[193]:


knn =  KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')


# In[194]:


knn.fit(X_train_std,y_train)


# # Predict KNN on the same data

# In[195]:


y_pred=knn.predict(X_test_std)


# In[196]:


(y_test != y_pred).sum()


# In[197]:


vis.plot_decision_regions(X=x_combined_std, y=y_combined, classifier = knn, test_idx=range(105,150))


# In[ ]:




