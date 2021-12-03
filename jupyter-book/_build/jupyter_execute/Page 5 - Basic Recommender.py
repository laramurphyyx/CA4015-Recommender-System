#!/usr/bin/env python
# coding: utf-8

# # Basic Recommender System

# ## Importing Relevant Packages and Datasets

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


dataframe_names = [
    'user_friends',
    'user_taggedartists',
    'artists',
    'tags',
    'user_artists',
    'user_artists_ratings']

file_names = [
    '../data/user_friends.csv',
    '../data/user_taggedartists.csv',
    '../data/artists.csv',
    '../data/tags.csv',
    '../data/user_artists.csv',
    '../data/user_artists_ratings.csv']


# In[3]:


for (dataframe, file) in zip(dataframe_names, file_names):
    vars()[dataframe] = pd.read_csv(file)


# ## Recommender System Outline

# For the basic recommender system, this will not take in any personalisation - it will be the same recommendations for all users.
# 
# This recommender system only recommends artists that have high ratings.
# 
# We will use the artists' average ratings, but this could be skewed by an artist who has very few ratings that yield a high average. To prevent this, we will take the top 10% of artists that have the highest number of user ratings, and then filter to find the artists with the highest average rating.
# 
# 
# 1. Make one for most popular all time
# 2. Make one for trending and use the timestamps one - maybe per genre?

# In[4]:


artists_users_ratings = (
    user_artists_ratings
    .groupby('artistID', as_index=False)
    .agg({'userID': ['count'], 'rating': ['mean']})
)


# In[5]:


artists_users_ratings.columns = ['artistID', 'count_users', 'avg_rating']


# In[6]:


plt.hist(artists_users_ratings['count_users'], bins=100)
plt.gca().set(
    title='Distribution of Users', 
    xlabel='Number of Users',
    ylabel='Frequency'
);


# In[7]:


plt.hist(artists_users_ratings['avg_rating'], bins=100)
plt.gca().set(
    title='Distribution of Average Rating', 
    xlabel='Average Rating',
    ylabel='Frequency'
);


# Make a comment about these graphs.

# ## Recommending the Top 5 Artists that You Might Like

# In[8]:


## Finding the top 10% of artists
np.percentile(artists_users_ratings['count_users'], 90)


# In[9]:


len(artists_users_ratings[artists_users_ratings['count_users']==1]['count_users'])/len(artists_users_ratings['count_users'])


# Over 60% of artists have only had one user listen to their music. 

# In[10]:


most_listened_to_artists = artists_users_ratings[artists_users_ratings['count_users']>=8]


# In[11]:


most_listened_to_artists['avg_rating'].max()


# Since the highest average rating for the top 10% of artists is only 3.5, we can either increase the amount of artists being considered (increase to the top 20% of artists), or we can identify the amount of times the artists is listened to per user. 

# ## Identifying Total Listens per User

# In[12]:


artists_total_listens = (
    user_artists_ratings
    .groupby('artistID', as_index=False)
    .agg({'userID': ['count'], 'rating': ['mean'], 'weight': ['sum']})
)


# In[13]:


artists_total_listens.columns = ['artistID', 'count_users', 'avg_rating', 'total_listens']


# In[14]:


plt.hist(artists_total_listens['total_listens'], bins=100)
plt.gca().set(
    title='Distribution of Total Listens', 
    xlabel='Number of Listens',
    ylabel='Frequency'
);


# In[15]:


## Finding the top 10% of artists
np.percentile(artists_total_listens['total_listens'], 90)


# In[16]:


artists_highest_total_listens = artists_total_listens[artists_total_listens['total_listens']>=4645]


# In[17]:


plt.hist(artists_highest_total_listens['avg_rating'], bins=20)

plt.gca().set(
    title='Distribution of Average Ratings for the top 10% Listened-To Artists', 
    xlabel='Number of Listens',
    ylabel='Frequency'
);


# In[18]:


artists_highest_listens = artists_highest_total_listens[artists_highest_total_listens['count_users']>=5]


# By filtering this dataset to only artists that have a user count above 5, removed artists that have one/two user(s) who have listened a very large number of times and have given 5-star ratings.

# In[19]:


plt.hist(artists_highest_listens['avg_rating']);


# In[20]:


recommendations = artists_highest_listens['avg_rating'].nlargest(2)


# In[21]:


recommendations


# In[22]:


artists_ordered_popularity = artists_highest_listens.sort_values(by='avg_rating', ascending=False)


# In[23]:


artist1_rec = artists_ordered_popularity.iloc[0,]['artistID']
artist2_rec = artists_ordered_popularity.iloc[1,]['artistID']


# In[24]:


artist1_details = artists[artists['id']==artist1_rec]
artist1_name = artist1_details.values[0][1]
artist1_link = artist1_details.values[0][2]


# In[25]:


artist2_details = artists[artists['id']==artist2_rec]
artist2_name = artist2_details.values[0][1]
artist2_link = artist2_details.values[0][2]


# In[26]:


artist2_link


# In[27]:


print("We think you might like " + artist1_name + ", and you can view their profile through this link: " + artist1_link)


# In[28]:


print("You might also like " + artist2_name + ", and you can view their profile through this link: " + artist2_link)


# It might also be interesting to allow the user to decide what percentage of artists they want to see

# Let's see the results for if we wanted 50% of the artists included.

# In[29]:


## Finding the top 10% of artists
np.percentile(artists_users_ratings['count_users'], 10)

