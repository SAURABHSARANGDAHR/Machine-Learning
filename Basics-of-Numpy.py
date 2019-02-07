#!/usr/bin/env python
# coding: utf-8

# # Array - https://goo.gl/njVmPU

# In[1]:


import numpy as np


# In[95]:


lst=[[1,2,3],
    [4,5,6]]
    
arr2d = np.array(lst)
arr2d


# In[7]:


arr2d.size


# In[8]:


arr2d.shape


# In[9]:


arr2d.itemsize


# In[12]:


arr2d.dtype


# In[13]:


arr_float = arr2d.astype(np.float32)
arr_float


# In[14]:


arr4x5 = [[1,2,3,4,5],
         [6,7,8,9,10],
         [11,12,13,14,15],
         [16,17,18,19,20]]
arr4x5=np.array(arr4x5,dtype='float32')


# In[15]:


arr4x5


# In[16]:


np.zeros((3,3))


# In[17]:


np.ones((2,4))


# In[18]:


np.eye(3)


# In[21]:


np.diag((8,8))


# In[30]:


np.random.rand(3,4)   #is used to generate a random matrix of specified dimensions


# In[31]:


np.arange(6)


# In[32]:


np.arange(45)


# In[33]:


np.arange(5,78)


# In[34]:


np.arange(-15,7)


# In[35]:


np.arange(4,15,2)   # (start,stop,step)


# In[36]:


np.arange(0,15,5)


# In[38]:


np.linspace(0.,1.,num=5)     #to divide the start and end values in equal no. of steps as mentioned in the 'num' variable


# In[39]:


np.arange(1,40,2)    # odd no. in array


# In[40]:


arr_ind = np.array([1,2,3])
arr_ind[0]


# In[41]:


arr_ind[2]


# In[42]:


arr_ind[:3]


# In[45]:


arr_ind[:2]       


# In[46]:


arr2d


# In[48]:


arr2d[0,:]


# In[49]:


arr2d[0,:1]


# In[50]:


arr2d[0,:2]


# In[51]:


arr2d[1,:2]


# In[52]:


arr4x5


# In[53]:


arr4x5[2,:]   # it means iy access all the columns


# In[54]:


arr4x5[:,2]   #in means it access all the rows


# In[55]:


arr4x5[:,:3]     #to access first three columns


# In[57]:


arr4x5[:2,:]


# In[58]:


arr4x5[:4,:4]  # here it will take 4 rows and 4 columns


# In[60]:


arr4x5[-2,:]   # using negative indexing


# In[61]:


arr4x5[-1,-1]


# In[63]:


arr4x5[:-2,:]


# In[64]:


arr4x5[:,-1]


# In[65]:


arr2d=np.array([[1,2,3],[4,5,6]])
print(np.add(arr2d,4))    # the original add is added with 4
arr2d   # the original array doesn't change


# In[67]:


print(np.subtract(arr2d,2))     # subtracts 2 from the original array


# In[68]:


arr2d*4


# In[69]:


arr4x5


# In[70]:


arr4x5*2


# In[71]:


arr4x5    # original doesn't change


# In[72]:


arr4x5**2


# In[74]:


arr4x5*.2


# In[75]:


arr4x5     # after all these operation the original array still remains the same


# In[79]:


newarr=np.array([[1,2,3],[4,5,6]])
newarr


# In[80]:


np.add.reduce(newarr)     # column wise addition


# In[81]:


np.add.reduce(newarr, axis=1)


# In[84]:


np.max(newarr)


# In[85]:


np.min(newarr)


# In[89]:


np.array([1,2,3]) + 1    # add every element by 1


# In[90]:


np.array([1,2,3]) - 1    # subtracts every element by 1


# In[101]:


arr1d = np.array([1,2,3,4,5,6])
arr1d


# In[102]:


arr2d_view = arr1d.reshape(2,3)
arr2d_view


# In[104]:


arr2d = np.array([[100,101,102],[103,104,105]])
arr2d


# In[111]:


arr2d.reshape(-1)


# In[112]:


arr2d


# In[113]:


arr2d.ravel()


# In[ ]:




