#!/usr/bin/env python
# coding: utf-8

# In[54]:


from keras.datasets import imdb


# In[55]:


from keras import preprocessing


# In[56]:


max_features = 10000 
maxlen = 20


# In[57]:


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words= max_features)


# In[58]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[59]:


x_train[0]


# In[60]:


len(x_train[0]), len(x_train[1])


# In[61]:


y_train[0], y_train[1]


# In[62]:


import numpy as np


# In[63]:


np.unique(y_train)


# In[64]:


x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)


# In[65]:


x_train.shape, x_test.shape


# In[66]:


x_train[0]


# ### MODEL

# In[67]:


from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding


# In[68]:


model=Sequential()
model.add(Embedding(10000,8, input_length = maxlen))


# In[69]:


model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))


# In[70]:


model.summary()


# ### Compile

# In[71]:


model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])


# ### Train

# In[72]:


history= model.fit(x_train,y_train, epochs = 10, batch_size= 32, validation_split=0.2 )


# ### Looking ata model history

# In[73]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss=history.history['val_loss']


# In[74]:


import matplotlib.pyplot as plt
epochs = range(1,len(acc)+1)


# In[75]:


plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title ("Training and Validation Accuracy")
plt.legend()
plt.figure()
plt.plot(epochs,loss, 'bo', label = 'Training loss')
plt.plot(epochs,val_loss, 'b', label = "Validation loss")
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[ ]:




