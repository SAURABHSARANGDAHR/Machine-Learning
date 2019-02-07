#!/usr/bin/env python
# coding: utf-8

# 
# 
# # Matplotlib Pyplot

# In[26]:


import matplotlib.pyplot as plt


# In[27]:


plt.plot([1,2,3,4])
plt.xlabel('X label')
plt.ylabel('Y label')
plt.show()


# In[28]:


plt.plot([1,2,3,4],[1,4,9,16], 'ys')
plt.axis([0,6,0,20])
plt.show()


# In[29]:


import numpy as np
t=np.arange(0.,5.,0.2)
plt.plot(t,t,'r--',t,t**2,'bs',t,t**3,'g^')
plt.show()


# In[30]:


import imageio
from skimage import transform


# In[31]:


im=imageio.imread('imageio:chelsea.png')
plt.imshow(im)
plt.show()


# In[32]:


im    # array of the pixel and the three digits shows the value of "RGB"


# In[33]:


im.shape      # it's a 300x451 array, 3 layers(RGB)


# In[34]:


lum_ing= im[:,:,0]
plt.imshow(lum_ing)
plt.show()


# In[35]:


lum_ing= im[:,:,0]
plt.imshow(lum_ing,cmap='hot')
plt.show()


# In[36]:


lum_ing= im[:,:,0]
plt.imshow(lum_ing,cmap='cool')
plt.show()


# In[37]:


lum_ing= im[:,:,0]
plt.imshow(lum_ing,cmap='hot')
plt.show()


# In[38]:


img=imageio.imread('imageio:chelsea.png')

img_next=transform.resize(img,(64,64))
pmgplot= plt.imshow(img_next)
plt.show()


# # IRIS FLOWER DATA SETS

# In[39]:


from sklearn import datasets


# In[40]:


iris = datasets.load_iris()


# In[41]:


iris


# In[42]:


print(iris.feature_names)


# In[43]:


iris.data.shape    # 150 samples 4 # data types 


# In[44]:


print(iris.data[:10])   # display top 10 of the data


# In[45]:


iris.target_names


# In[46]:


plt.plot(iris.data[:,:1],iris.data[:,1:2],'ro')
plt.show()


# In[47]:


plt.plot(iris.data[:,:1],iris.data[:,1:2],'g^')
plt.show()


# In[49]:


plt.plot(iris.data[:,:1],iris.data[:,1:2],'y^',iris.data[:,2:3],iris.data[:,3:4],'ro')
plt.show()


# In[ ]:




