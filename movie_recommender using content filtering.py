#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[11]:


#STEP 1:Read CSV from pandas library
df=pd.read_csv("movie_dataset.csv")
print(df.head())


# In[12]:


print(df.columns)


# In[13]:


#Select features
features=['keywords','cast','genres','director']

for feature in features:
    df[feature]=df[feature].fillna('') #fill all na with empty string
#Create a column in dataframe that combines all features

def combine_features(row):
    try:
        return row['keywords']+" "+row["cast"]+" "+row["genres"]+" "+row["director"]
    except:
        print("Error",row)

df["combined_features"]=df.apply(combine_features,axis=1) #for vertically combining
print("combined_features",df["combined_features"].head())


# In[14]:


#create count matrix from this new combined column
cv=CountVectorizer()
count_matrix=cv.fit_transform(df["combined_features"])


# In[15]:


#compute cosine similarity
cosine_sim=cosine_similarity(count_matrix)
movie_user_likes="Avatar"


# In[16]:


#helper functions
def get_title_from_index(index):
    return df[df.index==index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title==title]["index"].values[0]


# In[17]:


#get index of this movie from its title
movie_index=get_index_from_title(movie_user_likes)

similar_movies=list(enumerate(cosine_sim[movie_index]))
sorted_similar_movies=sorted(similar_movies,key=lambda x:x[1],reverse=True)


# In[19]:


#print titles of similar movies
i=0
for element in sorted_similar_movies:
		print (get_title_from_index(element[0]))
		i=i+1
		if i>50:
			break


# In[ ]:




