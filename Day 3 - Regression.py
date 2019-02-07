#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# # Z = 2X**2-3Y**2+1+e

# In[2]:


x = np.arange(-1,1, step=0.01)
y = np.arange(-1,1, step=0.01)


# In[3]:


len(x), len(y)


# In[4]:


X,Y=np.meshgrid(x,y)


# In[5]:


X.shape, Y.shape


# In[6]:


c=np.ones((200,200))


# In[7]:


e = np.random.rand(200,200)*0.1


# In[8]:


Z = 2*X*X - 3*Y*Y + 5*e


# In[9]:


Z.shape


# In[10]:


import vis


# In[11]:


vis.plot3d(X,Y,Z)


# In[12]:


X.shape, Y.shape, Z.shape


# In[13]:


input_xy= np.c_[X.reshape(-1),Y.reshape(-1)]
output_z=Z.reshape(-1)


# In[14]:


input_xy.shape, output_z.shape


# # Model

# In[15]:


from keras.models import Sequential
from keras.layers import Dense


# In[16]:


model= Sequential()


# In[19]:


model.add(Dense(40,input_dim = 2, activation= 'relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20, activation='relu'))

model.add(Dense(1))


# In[20]:


model.summary()


# In[29]:


model.compile(loss = 'mean_squared_error', optimizer='sgd')


# In[30]:


output = model.fit(input_xy, output_z, epochs=10, validation_split=0.1)


# # Predict

# In[31]:


z_pred = model.predict(input_xy).reshape((200,200))


# In[32]:


vis.plot3d(X,Y,z_pred)


# In[ ]:




