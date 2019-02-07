#!/usr/bin/env python
# coding: utf-8

# # Implement Standard Neural Net for Fashion MNIST

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist


# In[2]:


(X_train, y_train), (X_test, y_test)= fashion_mnist.load_data()


# In[3]:


X_train.shape


# In[4]:


X_train[0]


# In[5]:


plt.imshow(X_train[0])


# In[6]:


print('X_train shape', X_train.shape)     # 60000 samples
print('X_test shape', X_test.shape)       # 10000 test samples
print('y_train shape', y_train.shape)
print('y_test shape', y_test.shape)


# In[7]:


X_train = X_train.reshape(60000,784)
X_test = X_test.reshape(10000,784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# In[8]:


X_train/=255
X_test/=255


# In[9]:


print ('train matrix shape', X_train.shape)
print ('test matrix shape', X_test.shape)


# In[10]:


np.unique(y_train, return_counts = True)


# In[11]:


from keras.utils import np_utils


# In[12]:


n_classes=10


# In[26]:


y_orig=y_test
print('Shape before one-hot coding', y_train.shape)
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)
print('Shape after one-hot coding', y_train.shape)


# In[27]:


y_train[0]


# In[28]:


from keras.models import Sequential    # Keras has all neural networks
from keras.layers.core import Dense,Activation


# In[29]:


model =  Sequential()
model.add(Dense(100,input_shape=(784,)))    # input is 28x28=784
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))


# In[30]:


model.summary()


# In[31]:


model.compile(loss='categorical_crossentropy', metrics = ['accuracy'], optimizer='adam')


# In[32]:


history = model.fit(X_train, y_train, batch_size = 10, epochs = 5, validation_data= [X_test, y_test])
#  epoch run 60000 samples each time


# In[33]:


y_pred = model.predict_classes(X_test)


# In[34]:


y_pred.shape


# In[35]:


X_temp = X_test[0].reshape(28,28)    # sample[0] is tested amongst 10000 samples


# In[36]:


plt.imshow(X_temp)
plt.show()


# In[37]:


y_pred[0] 


# In[39]:


i_i=np.nonzero(y_pred != y_orig)[0]


# In[ ]:




