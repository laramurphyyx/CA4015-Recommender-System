#!/usr/bin/env python
# coding: utf-8

# # Data Exploration

# ## Importing Relevant Packages and Datasets

# In[1]:


import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt


# In[2]:


dataframe_names = [
    'user_friends',
    'user_taggedartists',
    'artists',
    'tags',
    'user_artists']

file_names = [
    '../data/user_friends.csv',
    '../data/user_taggedartists.csv',
    '../data/artists.csv',
    '../data/tags.csv',
    '../data/user_artists.csv']


# In[3]:


for (dataframe, file) in zip(dataframe_names, file_names):
    vars()[dataframe] = pd.read_csv(file)


# # Exploring the Users' Friends Dataset

# In[4]:


users_friends_count = (
    user_friends
    .groupby('userID', as_index=False)
    .agg({'friendID': ['count']})
)


# In[5]:


users_friends_count.columns = ['userID', 'count_friends']


# In[6]:


mean = str(users_friends_count['count_friends'].mean())
min = str(users_friends_count['count_friends'].min())
max = str(users_friends_count['count_friends'].max())

print('The average number of friends each user has is ' + mean + ' friends')
print('The user with the least amount of friends has ' + min + ' friends')
print('The user with the most amount of friends has ' + max + ' friends')


# In[7]:


data = users_friends_count['count_friends']
plt.hist(data, weights=np.ones(len(data)) / len(data), bins=120, edgecolor='black')
plt.gca().set(
    title='Distribution of Number of Friends per User', 
    xlabel='Number of Friends',
    ylabel='Frequency'
);


# As you can see in the above chart, over 60% of the users in this dataset have between 0 and 10 friends. Only ~15% of users have between 10 and 20 friends. 
# 
# We can also conclude that only less than 25% of users have over 20 friends. This is useful to know, as this would imply that a recommender system based off a user's friends' favourite artists alone may not perform accurately.
# 
# Although using friends' liked artists as a sole method to recommend songs will not perform well, it might be useful to put a slightly larger weight on their friends' artists, as long as they have a similar taste in music.

# # Exploring Users' Tagged Artists

# In[8]:


tagged_artists_users = (
    user_taggedartists
    .groupby('artistID', as_index=False)
    .agg({'userID': ['count']})
)

tagged_artists_users = (
    user_taggedartists
    .groupby('artistID', as_index=False)
    .agg({'userID': lambda x: x.nunique()})
)


# In[9]:


tagged_artists_users.columns = ['artistID', 'count_users']


# In[10]:


mean = str(tagged_artists_users['count_users'].mean())
min = str(tagged_artists_users['count_users'].min())
max = str(tagged_artists_users['count_users'].max())

print('The average artist has been assigned a tag from ' + mean + ' users')
print('The user with the least amount of of users assigning tags has ' + min + ' user(s)')
print('The user with the most amount users assigning tags has ' + max + ' users')


# In[11]:


data = tagged_artists_users['count_users']
plt.hist(data, weights=np.ones(len(data)) / len(data), bins=200)
plt.gca().set(
    title='Distribution of Number of Users assigning Tags per Artist', 
    xlabel='Number of Users',
    ylabel='Frequency'
);


# The plot above shows that there are very few artists that have over 50 users assigning tags.

# In[12]:


tagged_artists_users[tagged_artists_users['count_users']>=50].count()


# There are 55 artists who have been assigned tags from over 50 users. This suggests that there isn't a mistaken outlier, and that the data is just very sparse (just over 1%) after reaching 50 users.

# In[13]:


tagged_artists_tags = (
    user_taggedartists
    .groupby('artistID', as_index=False)
    .agg({'tagID': lambda x: x.nunique()})
)


# In[14]:


tagged_artists_tags.columns = ['artistID', 'count_tags']


# In[15]:


mean = str(tagged_artists_tags['count_tags'].mean())
min = str(tagged_artists_tags['count_tags'].min())
max = str(tagged_artists_tags['count_tags'].max())

print('The average artist has been assigned ' + mean + ' tags')
print('The user with the least amount of tags has ' + min + ' tag(s)')
print('The user with the most amount tags has ' + max + ' tags')


# In[16]:


data = tagged_artists_tags['count_tags']
plt.hist(data, weights=np.ones(len(data)) / len(data), bins=300)
plt.gca().set(
    title='Distribution of Number of Tags assigned to each Artist', 
    xlabel='Number of Tags',
    ylabel='Frequency'
);


# The above graph shows that over 25% of artists have received only 1 unique tag. 
# 
# The frequency of tags seems to be very close to 0 when the number of tags is greater than 75.

# In[17]:


tagged_artists_tags[tagged_artists_tags['count_tags']<=75].count()/len(tagged_artists_tags)


# In[18]:


tagged_artists_tags[tagged_artists_tags['count_tags']<=5].count()/len(tagged_artists_tags)


# Only 81 (0.65%) of the artists have greater than 75 unique tags assigned to them. 
# 
# In fact, over half of the artists have been assigned between 1 and 5 unique tags.
# 
# These tags will allow for a more complex recommendation to be made to a user.

# # Exploring the Artists and Tags Dataset

# In[19]:


artists.head()


# In[20]:


tags.head()


# The artists and tags datasets are just a reference table to expand on the artist details or the tag details. 
# 
# This does not require any more investigation, as there are no relevant data points to be used in our recommender system.

# ## Exploring Users' Liked Artists

# In[21]:


distribution_of_weights_given = (
    user_artists
    .groupby('weight', as_index=False)
    .agg({'userID': ['count']})
)


# In[22]:


distribution_of_weights_given.columns = ['weight', 'users_count']


# In[23]:


data = distribution_of_weights_given['users_count']
plt.hist(data, weights=np.ones(len(data)) / len(data), bins=200)
plt.gca().set(
    title='Distribution of Weights ', 
    xlabel='Number of Users',
    ylabel='Frequency'
);


# In[24]:


user_artists['weight']


# In[25]:


list_of_weights = np.asarray(user_artists['weight'])


# In[26]:


plt.hist(list_of_weights);


# In[27]:


list_of_weights.mean()


# In[28]:


plt.hist(list_of_weights, bins=300);


# Let's find out what proportion of people's weights are under 10,000

# In[29]:


user_artists[user_artists['weight']<=10000].count()/92834


# 99.32% of people are under 10,000 listens

# In[30]:


user_artists[user_artists['weight']<=2328].count()/92834


# 95% of the users listen to artists under 2328 times

# In[31]:


plt.hist(user_artists[user_artists['weight']<=2328]['weight'], bins = 30);

