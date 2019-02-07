#!/usr/bin/env python
# coding: utf-8

# In[3]:


from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions


# In[4]:


model = ResNet50(weights='imagenet')


# In[5]:


model.summary()


# In[39]:


img_path = 'mug.jpg'
img = image.load_img(img_path, target_size=(224,224))


# In[40]:


import numpy as np
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


# In[41]:


x.shape


# In[42]:


preds = model.predict(x)


# In[43]:


print('Predicted:', decode_predictions(preds, top=3)[0])


# In[ ]:




