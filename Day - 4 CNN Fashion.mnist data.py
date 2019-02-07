#!/usr/bin/env python
# coding: utf-8

# import numpy as np
# from keras.datasets import fashion_mnist

# In[52]:


(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


# In[53]:


X_train.shape


# In[54]:


y_train.shape


# In[55]:


X_test.shape, y_test.shape


# In[56]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('maplotlib', '')
import vis


# In[57]:


vis.fashion_mnist_label()


# In[58]:


plt.imshow(X_train[0])


# In[59]:


plt.imshow(X_train[0], cmap='Purples')


# In[60]:


y_train[0]


# In[61]:


number=155
plt.imshow(X_train[number], cmap="gray")


# In[62]:


y_train[155]


# In[63]:


plt.imshow(X_train[301], cmap='gray')


# In[64]:


y_train[301]


# In[65]:


X_train_conv = X_train.reshape(X_train.shape[0],28,28,1)
X_test_conv = X_test.reshape(X_test.shape[0],28,28,1)
X_train_conv.shape, X_test_conv.shape


# # Categorical Data

# In[66]:


from keras.utils import to_categorical


# In[67]:


y_train_class = to_categorical(y_train,10)
y_test_class = to_categorical(y_test,10)


# In[68]:


y_train_class.shape, y_test_class.shape


# # CNN Model

# In[69]:


from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout


# In[70]:


cnn = Sequential()


# In[71]:


cnn.add(Conv2D(32, kernel_size=(3,3), activation="relu",input_shape=(28,28,1)))


# In[72]:


cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(64, kernel_size=(3,3), activation="relu"))


# In[73]:


cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Flatten())


# In[74]:


cnn.add(Dense(128, activation="relu"))


# In[75]:


cnn.add(Dropout(0.25))
cnn.add(Dense(10, activation="softmax"))


# In[76]:


cnn.summary()


# # Train

# In[77]:


cnn.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=['accuracy'])


# In[79]:


ouput_cnn = cnn.fit(X_train_conv, y_train_class, batch_size= 128, epochs=10, validation_data=(X_test_conv,y_test_class))


# # Let's Predict

# In[91]:


y_pred= cnn.predict_classes(X_test_conv)


# In[92]:


y_pred.shape


# In[101]:


X_temp=X_test[2]


# In[102]:


plt.imshow(X_temp)


# In[103]:


y_pred[2]


# In[114]:


i_i = np.nonzero(y_pred != y_test_class )


# In[111]:


len(i_i)


# In[112]:


i_i


# In[ ]:




