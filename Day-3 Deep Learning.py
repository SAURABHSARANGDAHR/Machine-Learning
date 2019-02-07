#!/usr/bin/env python
# coding: utf-8

# # MNIST

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist     # MNIST is a large database which consist over 60000 images of handwritten digits


# In[2]:


(X_train, y_train),(X_test, y_test)= mnist.load_data()   # Loading the data in the 2 variable


# In[3]:


X_train.shape


# In[4]:


X_train[0]    # '0' th position of the array


# In[5]:


X_train[0].shape     # it is a 28x28 pixel image


# In[6]:


plt.imshow(X_train[1000],cmap='gray', interpolation ='none')   # you select any no. from 0 to 59999 as there are 60000 images.
plt.show()


# In[7]:


plt.imshow(X_train[1010],cmap='gray', interpolation ='none')
plt.show()


# In[8]:


plt.imshow(X_train[100],cmap='gray', interpolation ='none')
plt.show()


# In[9]:


plt.imshow(X_train[1001],cmap='gray', interpolation ='none')
plt.show()


# In[10]:


plt.imshow(X_train[59999],cmap='gray', interpolation ='none')
plt.show()


# In[11]:


plt.imshow(X_train[344],cmap='gray', interpolation ='none')
plt.show()


# In[12]:


plt.imshow(X_train[2598],cmap='cool', interpolation ='none')
plt.show()


# In[13]:


plt.imshow(X_train[3494],cmap='hot', interpolation ='none')
plt.show()


# In[14]:


import vis


# In[15]:


vis.imshow_sprite(X_train[:100])    # first 100 images


# In[16]:


28*28


# In[17]:


print('X_train shape', X_train.shape)     # 60000 samples
print('X_test shape', X_test.shape)       # 10000 test samples
print('y_train shape', y_train.shape)
print('y_test shape', y_test.shape)


# # Build input vector from 28x28 pixel

# In[18]:


X_train = X_train.reshape(60000,784)
X_test = X_test.reshape(10000,784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# # Normalize

# In[19]:


X_train/=255
X_test/=255


# # Print shape of final data

# In[20]:


print ('train matrix shape', X_train.shape)
print ('test matrix shape', X_test.shape)


# In[21]:


np.unique(y_train, return_counts = True)


# In[22]:


from keras.utils import np_utils


# # one-hot encoding

# In[23]:


n_classes=10


# In[24]:


y_orig= y_test
print('Shape before one-hot coding', y_train.shape)
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)
print('Shape after one-hot coding', y_train.shape)


# In[25]:


y_train[0]


# # Keras Neural Network

# In[26]:


from keras.models import Sequential    # Keras has all neural networks
from keras.layers.core import Dense,Activation


# In[27]:


model =  Sequential()
model.add(Dense(512,input_shape=(784,)))    # input is 28x28=784
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))


# In[28]:


model.summary()    # Param is 401920... for o/p: (401920x60000xno. of epochs) is the calculation done


# # Compile Model

# In[29]:


model.compile(loss='categorical_crossentropy', metrics = ['accuracy'], optimizer='adam')


# # Train

# In[30]:


history = model.fit(X_train, y_train, batch_size = 10, epochs = 5, validation_data= [X_test, y_test])
#  epoch run 60000 samples each time


# # Lets Predict

# In[31]:


# we haven't used 10000 test samples. Here's it's prediction


# In[32]:


y_pred = model.predict_classes(X_test)


# In[33]:


y_pred.shape


# In[34]:


X_temp = X_test[0].reshape(28,28)    # sample[0] is tested amongst 10000 samples


# In[35]:


plt.imshow(X_temp)
plt.show()


# In[36]:


y_pred[0]   # the o/p given by ANN is '7' which is correct 


# In[37]:


X_temp = X_test[9875].reshape(28,28)
plt.imshow(X_temp)
plt.show()


# In[38]:


y_pred[9875]


# In[39]:


X_temp = X_test[9999].reshape(28,28)
plt.imshow(X_temp)
plt.show()


# In[40]:


y_pred[9999]


# In[41]:


X_temp = X_test[3].reshape(28,28)
plt.imshow(X_temp)
plt.show()


# In[42]:


y_pred[3]


# In[43]:


X_temp = X_test[78].reshape(28,28)
plt.imshow(X_temp)
plt.show()


# In[44]:


y_pred[78]


# In[45]:


X_temp = X_test[115].reshape(28,28)
plt.imshow(X_temp)
plt.show()


# In[46]:


y_pred[115]


# In[47]:


X_temp = X_test[150].reshape(28,28)
plt.imshow(X_temp)
plt.show()


# In[48]:


y_pred[150]


# In[55]:


i_i = np.nonzero(y_pred != y_orig)[0]   # we wrote nonzero meaning the multidimension array doesn't show zero arrays 
                                        # we wrote[0]at the end to show single dimension as numpy works on multidimension, 
                                        # this way, other dimension arrays aren't displayed, whose values are anyway [0]


# In[56]:


len(i_i)


# In[57]:


i_i #  these are the error positions that have occured


# In[58]:


i_i.shape


# In[59]:


X_temp = X_test[151].reshape(28,28)
plt.imshow(X_temp)
plt.show()


# In[61]:


y_pred[151]


# In[ ]:


s

