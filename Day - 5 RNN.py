#!/usr/bin/env python
# coding: utf-8

# In[6]:


from keras.datasets import imdb
from keras.preprocessing import sequence


# In[7]:


max_features=10000
maxlen = 500


# In[11]:


(input_train, y_train), (input_test, y_test)= imdb.load_data(num_words=max_features)


# In[12]:


input_train.shape, y_train.shape


# In[14]:


X_train = sequence.pad_sequences(input_train, maxlen = maxlen)
X_test = sequence.pad_sequences(input_test, maxlen = maxlen)


# In[15]:


X_train.shape, X_test.shape


# # RNN

# In[16]:


from keras.models import Sequential
from keras.layers import Dense, Embedding, SimpleRNN


# In[18]:


model=Sequential()
model.add(Embedding(max_features,32))
model.add(SimpleRNN(32))
model.add(Dense(1,activation='sigmoid'))


# In[19]:


model.summary()


# In[21]:


model.compile(optimizer = 'rmsprop', loss= 'binary_crossentropy', metrics = ["acc"])


# In[23]:


history= model.fit(X_train,y_train, epochs = 10, batch_size= 128, validation_split=0.2 )


# In[24]:


from keras.layers import LSTM


# In[26]:


model_lstm = Sequential()
model_lstm.add(Embedding(max_features,32))
model_lstm.add(LSTM(32))
model_lstm.add(Dense(1, activation='sigmoid'))


# In[27]:


model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])


# In[29]:


history= model.fit(X_train,y_train, epochs = 10, batch_size= 128, validation_split=0.2 )


# In[ ]:




