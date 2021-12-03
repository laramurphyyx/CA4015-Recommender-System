#!/usr/bin/env python
# coding: utf-8

# # Assigning Tags to Artists

# In[1]:


import pandas as pd
import numpy as np
import heapq


# In[2]:


artists = pd.read_csv('../data/artists.csv')
user_taggedartists = pd.read_csv('../data/user_taggedartists.csv')
tags = pd.read_table('../data/tags.dat', sep="\t", encoding='latin-1')
user_artists_ratings = pd.read_csv('../data/user_artists_ratings.csv')


# ## Finding the top 2 tags per artist

# In[3]:


artists['first_tag'] = 0
artists['second_tag'] = 0


# In[4]:


for artist in user_taggedartists.artistID.unique():
    artist_data = user_taggedartists[user_taggedartists['artistID']==artist]
    if len(artist_data) < 1:
        first_tag = 0
        second_tag = 0
    else:
        top_two_tags = artist_data['tagID'].value_counts()[:3].index.tolist()
        first_tag = top_two_tags[0]
        if len(top_two_tags)>1:
            second_tag = top_two_tags[1]
        else:
            second_tag = 0
    artists.loc[artists.id == artist, 'first_tag'] = first_tag
    artists.loc[artists.id == artist, 'second_tag'] = second_tag


# In[5]:


artists.to_csv('../data/artists_tags.csv', index=False)

