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


# In[52]:


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


# In[53]:


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


# In[85]:


# 4 optional

def getCityAvgTemp(city_name, data):
    return [data[1][i] for i, city in enumerate(data[3]) if city == city_name]

Rome = getCityAvgTemp("Rome", dataTemp)
Bangkok = getCityAvgTemp("Bangkok", dataTemp)

plt.hist(Rome, label='Rome')
plt.hist(Bangkok, label='Bangkok')
plt.title('Distribution of the average land temperatures for Rome and Bangkok')
plt.legend()
plt.show()


# They are too high, maybe it was used a *Fahrenheit* scale

# In[91]:


# 5 optional

def faToCel(data):
    return list(map(lambda x: (x-32)/1.8, Bangkok))    # Fahrenheit to Celsius 

BangkokCelsius = faToCel(Bangkok)

plt.hist(Rome, label='Rome')
plt.hist(BangkokCelsius, label='Bangkok')
plt.title('Distribution of the average land temperatures for Rome and Bangkok')
plt.legend()
plt.show()


# ## Exercise 2.2

# In[3]:


# 1

IMDB = [[],[]]

with open('./aclimdb_reviews_train.txt') as f:
    header = f.readline()
    for row in csv.reader(f):
        if len(row) == 2:
            for i in range(2):
                if i == 1:
                    IMDB[i].append(int(row[i]))
                else:
                    IMDB[i].append(row[i])


# In[4]:


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


# In[5]:


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


# In[6]:


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


# In[7]:


# 5

def TFIDF(tf, idf):
    result = []
    for doc in tf:
        doc_tfidf = {word:(doc[word]*idf[word]) for word in doc.keys()} 
        result.append(doc_tfidf)
    return result

tdidf_tokens = TFIDF(tf_tokens, idf_tokens)


# In[31]:


# 6 optional

def clusterize(data, tdidf_data, train_index):    # Divide data into positive and negative comments
    pos = []
    neg = []
    for i, score  in enumerate(IMDB[1]):
        if i == train_index:    # the first document is for testing
            continue
        if score == 1:
            pos.append(tdidf_data[i])
        else:
            neg.append(tdidf_data[i])
    return pos, neg


def norm(d):
    """Compute the L2-norm of a vector representation."""
    return sum([ tf_idf**2 for t, tf_idf in d.items() ])**.5

def dot_product(d1, d2):
    """Compute the dot product between two vector representations."""
    word_set = set(list(d1.keys()) + list(d2.keys()))
    return sum([(d1.get(d, 0.0) * d2.get(d, 0.0)) for d in word_set])

def cosine_similarity(d1, d2):
    """
    Compute the cosine similarity between documents d1 and d2.
    Input: two dictionaries representing the TF-IDF vectors for documents d1 and d2.
    Output: the cosine similarity.
    """
    return dot_product(d1, d2) / (norm(d1) * norm(d2))

def mean_similarity(cos_sim):
    return sum(elem for elem in cos_sim)/len(cos_sim)


test_document = tdidf_tokens[0]

positive, negative = clusterize(IMDB, tdidf_tokens, 0)

cosine_pos = []
cosine_neg = []

for elem in positive:
    cosine_pos.append(cosine_similarity(elem, test_document))
    
for elem in negative:
    cosine_neg.append(cosine_similarity(elem, test_document))
    
pos_mean = mean_similarity(cosine_pos)
neg_mean = mean_similarity(cosine_neg)

print(f"Mean of the cosine similarity in the positive group: {pos_mean}")
print(f"Mean of the cosine similarity in the negative group: {neg_mean}")

print("The 2 cosines are too close to discriminate")

