#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.preprocessing.text import Tokenizer


# In[2]:


sample = {"The cat sat on the mat.", "The dog ate my homework."}


# In[16]:


tokenizer = Tokenizer(num_words=10)


# In[17]:


tokenizer.fit_on_texts(sample)


# In[18]:


sequences = tokenizer.texts_to_sequences(sample)


# In[19]:


sequences


# ### One hot encoding

# In[20]:


one_hot_results = tokenizer.texts_to_matrix(sample, mode='binary')


# In[21]:


one_hot_results


# In[22]:


one_hot_counts = tokenizer.texts_to_matrix(sample, mode = 'count')


# In[24]:


one_hot_counts


# In[26]:


one_hot_tfidf = tokenizer.texts_to_matrix(sample, mode = "tfidf")


# In[27]:


one_hot_tfidf


# In[28]:


word_index = tokenizer.word_index


# In[29]:


word_index


# In[ ]:




