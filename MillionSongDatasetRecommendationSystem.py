#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv(r"C:\Users\ASUS\Downloads\spotify_millsongdata.csv")


# In[3]:


df.head(2)


# In[4]:


df['text'].head(1).values


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[7]:


df.duplicated().any()


# In[8]:


df=df.drop(columns=['link'])


# In[9]:


df


# In[10]:


df['text']=df['text'].apply(lambda x: x.lower())
    


# In[11]:


df


# Text Preprocessing
# 

# In[12]:


import regex


# In[13]:


type(df['text'].head(1).values)


# In[14]:


df['text']=df['text'].replace(r'^\w','')


# In[15]:


df


# In[16]:


df


# In[17]:


df['text'].head(1).values


# In[18]:


df['text'].replace(r'\r\n','')


# In[19]:


df['text'] = df['text'].str.lower().replace(r'^\w\s', '').replace(r'\n', '', regex = True)


# In[20]:


df['text'].head(1).values


# In[21]:


df['text']=df['text'].replace(r'\r','',regex=True)


# In[22]:


df['text'].sample(1).values


# In[23]:


import nltk


# In[24]:


from nltk.tokenize import word_tokenize


# In[25]:


df['text']=df['text'].apply(lambda x: word_tokenize(x))


# In[26]:


df=df.head(10000)


# In[27]:


from nltk.stem.porter import PorterStemmer


# In[28]:


s=PorterStemmer()


# In[29]:


from nltk.corpus import stopwords


# In[30]:


k=stopwords.words('english')


# In[31]:


k


# In[32]:


import string
p=string.punctuation


# In[33]:


p


# In[34]:


def tags(x):
    list=[]
    for i in x:
        if i not in k:
            if i not in p:
               list.append(i)
                
    return list
                


# In[35]:


df['text']=df['text'].apply(tags)


# In[36]:


df


# In[37]:


df['text']=df['text'].apply(lambda x: [s.stem(i) for i in x])


# In[38]:


from sklearn.feature_extraction.text import CountVectorizer


# In[39]:


from sklearn.metrics.pairwise import cosine_similarity


# In[40]:


cv=CountVectorizer(max_features=100000)


# In[41]:


df['text']=df['text'].apply(lambda x: ' '.join(x))


# In[42]:


df


# In[43]:


vectors=cv.fit_transform(df['text']).toarray()


# In[44]:


cv


# In[45]:


vectors


# In[46]:


distances=cosine_similarity(vectors)


# In[47]:


d=distances[0]


# In[57]:


p=sorted(list(enumerate(d)),reverse=True,key=lambda x:x[1])[1:11]


# In[58]:


p


# In[59]:


df.shape


# In[60]:


df.iloc[9634].song


# In[66]:


for i in p:
    print(df.iloc[i[0]].song)


# In[78]:


def recommend(x):
    index=df[df['song']==x].index[0]
    dist=distances[index]
    p=sorted(list(enumerate(dist)),reverse=True,key=lambda x:x[1])[1:11]
    
    for i in p:
        print(df.iloc[i[0]].song)


# In[79]:


recommend('Once More')


# In[ ]:




