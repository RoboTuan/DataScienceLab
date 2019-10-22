#!/usr/bin/env python
# coding: utf-8

# ## Preliminaries 

# In[1]:


from random import gauss
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

l = [gauss(0, 1) for _ in range(500)]
plt.hist(l)
plt.title('Gaussian distribution (mu=0, sigma=1)')
plt.show()


# ## Exercize 2.1

# 
# • Date  
# • AverageTemperature  
# • AverageTemperatureUncertainty  
# • City  
# • Latitude  
# • Longitude  

# In[2]:


# 1

import csv


dataTemp = [[],[],[],[],[],[],[]]

with open("./GLT_filtered.csv") as f:
    header = f.readline()
    #print(header)
    for row in csv.reader(f):
        if len(row) == 7:
            for i in range(7):
                #print(row)
                if i == 1 and row[i] != '':
                    dataTemp[i].append(float(row[i]))
                else:
                    dataTemp[i].append(row[i])


# In[3]:


# 2

for i, number in enumerate(dataTemp[1]):
    if number == '':
        if i == 0:
            for n in range(i+1, len(dataTemp[1])):
                if dataTemp[1][n] != '':
                    number = dataTemp[1][n]/2
                    break
        elif i == len(dataTemp[1]):
            for n in range(i-1, 0, -1):
                if dataTemp[1][n] != '':
                    number = dataTemp[1][n]/2
                    break
        else :
            sum = 0
            for n in range(i+1, len(dataTemp[1])):
                if dataTemp[1][n] != '':
                    sum += dataTemp[1][n]
                    break
            for n in range(i-1, -1, -1):
                if dataTemp[1][n] != '':
                    sum += dataTemp[1][n]
                    break
            dataTemp[1][i] = sum/2


# In[4]:


# 3

def getHotCold(city_name, data, N):
    measurements = [data[1][i] for i, city in enumerate(data[3]) if city == city_name]
    
    N_hot = [measurement for i, measurement in enumerate(sorted(measurements, reverse = True)) if i < N]
    N_cold = [measurement for i, measurement in enumerate(sorted(measurements)) if i < N]
    
    return N_hot, N_cold
    
a,b = getHotCold("Abidjan", dataTemp, 10)

print("Top 10 hottest measurements for Abidjan:")
[print(m) for m in a]
print()
print("Top 10 coldest measurements for Abidjan:")
[print(m) for m in b]
print()


# ## Exercise 2.2

# In[5]:


# 1

IMDB = [[],[]]

with open('./aclimdb_reviews_train.txt') as f:
    header = f.readline()
    for row in csv.reader(f):
        if len(row) == 2:
            for i in range(2):
                IMDB[i].append(row[i])


# In[6]:


# 2

import string

def tokenize(docs):
    """Compute the tokens for each document.
    
    Input: a list of strings. Each item is a document to tokenize.
    Output: a list of lists. Each item is a list containing the tokens of the relative document.
    """
    tokens = []
    for doc in docs:
        for punct in string.punctuation:
            doc = doc.replace(punct, " ")
        split_doc = [ token.lower() for token in doc.split(" ") if token ]
        tokens.append(split_doc)
    return tokens


IMDB_tokens = tokenize(IMDB[0])


# In[7]:


# 3


def TF(docs): 
    result = []
    for doc in docs:
        doc_tf = {}
        
        for word in doc:  
            if word not in doc_tf:
                doc_tf[word] = 1
            else :
                doc_tf[word] += 1
        result.append(doc_tf)

    return result

tf_tokens = TF(IMDB_tokens)


# In[16]:


# 4

from math import log

n_docs = len(IMDB_tokens)
words = {word:0 for word in set([word for doc in IMDB_tokens for word in doc])}


def DF(docs, dictionary):
    for doc in docs:
        for word in doc:
            dictionary[word] += 1  
    return dictionary

df_tokens = DF(IMDB_tokens,words)

idf_tokens = {k:log(n_docs/v) if log(n_docs/v) >= 0 else 0 for k, v in df_tokens.items()}

idf_ascending = [k for k,v in sorted(idf_tokens.items(), key = lambda x: x[1])] 

# the most used words are the most common in english language such as:
# prepositions, adverbs, articles, versbs an domain specific names (like movie, film, ecc..)


# In[20]:


# 5

def TFIDF(tf, idf):
    result = []
    for doc in tf:
        doc_tfidf = {word:(doc[word]*idf[word]) for word in doc.keys()} 
        result.append(doc_tfidf)
    return result

tdidf_tokens = TFIDF(tf_tokens, idf_tokens)


# In[ ]:




