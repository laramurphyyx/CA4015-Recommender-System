#!/usr/bin/env python
# coding: utf-8

# # Content-Based Recommender System

# ## Importing Relevant Packages and Datasets

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


dataframe_names = [
    'user_friends',
    'user_taggedartists',
    'artists',
    'tags',
    'user_artists',
    'user_artists_ratings',
    'artists_tags']

file_names = [
    '../data/user_friends.csv',
    '../data/user_taggedartists.csv',
    '../data/artists.csv',
    '../data/tags.csv',
    '../data/user_artists.csv',
    '../data/user_artists_ratings.csv',
    '../data/artists_tags.csv']


# In[3]:


for (dataframe, file) in zip(dataframe_names, file_names):
    vars()[dataframe] = pd.read_csv(file)


# ## Adjusting Existing Columns

# Importing artists that have their top two tags and converting them to their string equivalents

# In[4]:


tag_dict = {}
tag_dict[0] = ""
for row in tags.itertuples():
    tagID = row[1]
    tag = row[2]
    tag_dict[tagID] = tag


# In[5]:


artists_tags['first_tag'] = artists_tags.apply(lambda row: tag_dict[row['first_tag']], axis=1)
artists_tags['second_tag'] = artists_tags.apply(lambda row: tag_dict[row['second_tag']], axis=1)


# ## Recommender Based on Top Tag

# We aim to calculate the Term Frequency-Inverse Document Frequency (TF-IDF) for each artist. The terms will be taken from the assigned tags for each artist.

# In[6]:


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(artists_tags['first_tag'])
tfidf_matrix.shape


# This 'tfidf' object was created to remove any stop words that may appear in the user-assigned tags for each artist.
# 
# This was then fitted as a matrix with the dimensions (N x M) where N is the number of artists and M is the number of tokens/words (excluding stop words) that appear in their top tag. In this case, there are 17,632 artists and 1,246 words.

# In[7]:


tfidf.get_feature_names()[900:910]


# This is an example of some of the tokens that appear in the top five tags of each artist. 
# 
# There are some tags that wouldn't be classified as genres, such as 'pleasures' or 'play', although these are unlikely to affect the performance of the recommender system.

# In[8]:


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# As we are using the TF-IDF Vectoriser, we can calculate the dot product between each vector to give us the cosine similarity score. This is more efficient than to use cosine_similarities().

# In[84]:


cosine_sim.shape


# This makes sense as this similarity is comparing each artist to each other, so each artist has it's own column and it's own row.

# In[132]:


cosine_sim[2]


# This is an example of the cosine similarities for the artist with the ID 2.
# 
# The artist will have 100% similarity with itself, hence there is 1.0 in the 2nd column.

# In[ ]:


indices = pd.Series(artists_tags.index, index=artists_tags['name']).drop_duplicates()


# The above 'indices' series is created to reverse the mapping of artist names and IDs, this allows for easier searching in the recommender system.

# In[141]:


def get_recommendations(name, cosine_sim=cosine_sim):
    # Get the index of the artist that matches the name
    index = indices[name]

    # Get the pairwsie similarity scores of all artists with that artist
    sim_scores = list(enumerate(cosine_sim[index]))

    # Sort the artists based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar artists
    sim_scores = sim_scores[1:11]

    # Get the artist indices
    artist_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return artists_tags['name'].iloc[artist_indices]


# The above function retrieves the top 10 recommendations based on the most common tag that each artist received.
# 
# This will work quite well for artists that have received lots of the same tag, making it more accurate, and may not perform as well on artists with few/no tags.

# In[150]:


get_recommendations('Kanye West')


# In[153]:


artists_tags[artists_tags['name']=='Flagelo Urbano'] ## hip hop under, raptuga, underground hip hp, hip hop
artists_tags[artists_tags['name']=='Common'] ## hip-hop, hip hop, rap, mean, 90s
artists_tags[artists_tags['name']=='Nach'] ## hip-hop


# In[144]:


get_recommendations('Coptic Rain')


# In[149]:


artists_tags[artists_tags['name']=='Diary of Dreams'] ## darkwave, german, gothic, seen live, industrial
artists_tags[artists_tags['name']=='Carpathian Forest'] ## black metal, true norwegian black metal, norwegian
artists_tags[artists_tags['name']=='Moi dix Mois'] ## j-rock, japanese, visual kei, gothic metal


# As you can see when we run 'Kanye West' into the recommender system, we are given lots of artist that are related to hip-hop or rap. Although, the third recommendation, Nach, only has one tag assigned to it, which suggests that Nach received very few tags, and may not be similar to Kanye West.
# 
# With our second example, we ran 'Coptic Rain' into the recommender system, to see how it would perform on an artist that had no tags. The tags of the top three recommended artists seem to be quite random (darkwave, german, norwegian, japanese). The system performed quite poorly for this artist.

# # Using the Top-5 Instead of Top-1 Tags

# In[ ]:


artists_tags['top_five_tags'] = artists_tags['top_five_tags'].fillna('')


# In[110]:


def find_name_and_tags(dataframe):
    return "".join(dataframe['name'].split()) + ' ' + dataframe['top_five_tags']

artists_tags['name_and_tags'] = artists_tags.apply(find_name_and_tags, axis=1)


# The name of the artist may be important to use in the recommender system as it may be referenced within some of the tags.
# 
# Some of the tags produced by the users may contain comments such as "sounds like michael jackson", and in this case it can create relations between Michael Jackson and these comments. It can also build relationships where the artist name may appear as "Jay-Z ft. Kanye West", so that Jay-Z and Kanye West could achieve a larger cosine similarity.
# 
# We are merging the name and the tags into one column to simplify things.

# In[107]:


count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(artists_tags['name_and_tags'])


# In this case, we use CountVectorizer() instead of TF-IDF, as previously done. This is just to avoid down-weighting important tags that may appear more often than others.

# In[113]:


count_matrix.shape


# There are much more words being produced when we accept the top 5 tags and the name of the artists. 

# In[ ]:


cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


# In[117]:


indices = pd.Series(artists_tags.index, index=artists_tags['name'])


# Here, we are creating the new cosine similarity matrix and the new indices object that can be passed into our original get_recommendations() function.

# In[118]:


get_recommendations('Kanye West', cosine_sim2)


# In[154]:


get_recommendations('Coptic Rain', cosine_sim2)


# When we run 'Kanye West' in the new recommender system, it seems to output more accurate results, the top 10 similar artists seem to be more similar this time. 
# 
# However, when we run 'Coptic Rain', who has no tags, we get other artists with very similar names, as it is the only token being considered. 
