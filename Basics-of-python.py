#!/usr/bin/env python
# coding: utf-8

# In[6]:


print ('Hello RAIT')


# In[8]:


x=6
type(x)


# In[11]:


x,y,z=4,5.6,'RAIT'
z


# ### width=17
# height=12.9
# delimeter='#'

# In[13]:


width=17
width/2


# In[14]:


height=12.9
(width+2)/height


# In[15]:


width+2/height


# In[16]:


delemeter='#'
delemeter*5


# # Raduis of sphere
# 4/3 pi r**3 

# In[17]:


r=5
vol= 4/3*3.14*5**3
vol


# # Conditional Statements

# In[22]:


x=12
if x>0:
    a="positive"
else:
    a='negative'
a


# In[29]:


word='This is a hands on workshop on machine learning and deeep learning'
count=0
for letter in word:
    if letter == 'g':
        count=count+1
count


# In[30]:


n=20
while n!=1:
    print(n)
    if n%2==0:
        n=n/2
    else:
        n=n*3+1


# In[8]:


def countdown(count):
    count=int(input("Enter a no.: "))
    while count>=0:
        print(count)
        count=count-1
    print('good bye')
countdown(count)


# In[2]:


fruit=' banana'
letter=fruit[4]
letter


# In[3]:


len(fruit)


# In[4]:


for i in fruit:
    print (i)


# In[5]:


fruit[::-1]*2


# In[6]:


greeting='Hello, world!'
greeting[0]


# In[9]:


le_str= 'This is a workshop on ML in RAIT'
words=le_str.split()
len(words)


# In[10]:


for a in words:
    print(a)


# In[12]:


le_str.find('a')


# In[13]:


le_str.find('in')


# In[16]:


le_str[-1:-4:-1]


# In[17]:


le_str.find('AIT')


# In[19]:


le_str[29:32:1]


# In[20]:


alist=[15,2,34,54.7,65.0]
alist[2]


# In[24]:


nested_list=[[1,2],[3,4,5],[6,7],[8],['this is a string']]
nested_list[4]


# In[25]:


nested_list[:2]


# In[26]:


nested_list[1:2]


# In[33]:


i=5
for i in range(len(nested_list)):
    print(nested_list[i])
    i+=1


# In[61]:


a=[1,2,3]
b=[4,5,7]
(a+b)


# In[62]:


c=[0]*5
c


# In[63]:


c=[0,0,0]*5
c


# In[72]:


b.insert(2,8.5,)        # here 2 is the index and 8.5 is the element to be inserted
b


# In[68]:


a=[1,2,3]


# In[74]:


b=[5,6]
a.extend(b)
a


# In[77]:


a.sort()
a


# In[79]:


a=[1,2,3,4,5,6,7,8,9]


# In[80]:


a.pop(4)


# In[83]:


p=[5,5,5]
q=[6,6,6]
p[0]=q[2]
p


# In[84]:


' '.join(['this','is','a','sentence'])


# In[96]:


' (0_o) '.join(['this','is','a','sentence'])


# In[103]:


eng2sp={'one':'uno','two':'dos','three':'tres'}
eng2sp['two']


# In[104]:


eng2sp['four']='quatro'
eng2sp['five']='cinco'
eng2sp['six']='seis'
eng2sp


# In[105]:


len(eng2sp)


# In[112]:


x=input('Enter the no. u want to convert: ')
if x in eng2sp:
    print(eng2sp['six'])
else:
    print('no key')


# In[113]:


eng2sp.items()


# In[115]:


eng2sp.keys()


# In[116]:


country={'India':['Delhi','Mumbai','Banglore'],'USA':['New York','Chicago','Boston']}
country


# In[117]:


country['India'][1]


# In[118]:


country['India'].append('Chennai')


# In[119]:


country


# In[120]:


country['USA'][2]


# In[121]:


country['USA'].append('Las Vegas')


# In[122]:


country


# In[124]:


t=('a','b','c','d','e')
t


# In[125]:


t1=tuple('MUMBAI')
t1


# In[127]:


t2=('apple','banana','cherry','apple','banana')
if 'apple' in t2:
    print('yes')


# In[130]:


t2.count('apple')


# In[132]:


t2.index('apple')


# In[ ]:




