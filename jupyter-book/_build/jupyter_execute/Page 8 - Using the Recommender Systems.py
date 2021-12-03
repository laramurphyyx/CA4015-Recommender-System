#!/usr/bin/env python
# coding: utf-8

# # Using the Recommender Systems

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from basic_recommender_system import *
from content_based_recommender import *


# These recommender systems can be used if you create a local version of this repo and change some of the variables in this noetbook.

# # Guideline to use the Basic Recommender System

# The <b>basic recommender system</b> can be called using the function Find_The_Most_Popular_Music().
# The parameters are as follows:
# 1. method : This is a string value, and can be either:
#     * most_users (default): Returns the two artists with the highest number of users
#     * most_listens : Takes the top 10% of artists with the highest user count and returns the two with the highest number of listens
#     * star_rating : Takes the top 10% of artists with the highest listens and removes artists with less than 5 users and returns the two artists with the highest average rating
#     * custom : This is a customised option for recommendations. You get to decide how to filter the artists (top 10% of artists with the highest user count, top 25% of users with the highest listen count, top 5% of users with the highest star-ratings, etc.). It also allows you to choose the final basis for the recommendation (user_count, listen_count or star_rating). It allows for the following optional arguments:
# 
# 
# 2. (optional) percentage_of_artists : This defaults to 10%. It takes the top 10% of artists based on your choice of filtering.
# 
# 
# 3. (optional) percentage_basis : This can be any of the following:
#     * user_count (default) : This takes the top percentage of artists based on the number of users listening to them
#     * listen_count : This takes the top percentage of artists based on the number of times they've been listened to
#     * star_rating : This takes the top percentage of artists based on their average star-rating
# 
# 
# 4. (optional) recommendation_basis : This can be any of the following:
#     * user_count : This returns the top two artists based on the number of users listening to them
#     * listen_count : This takes the top two artists based on the number of times they've been listened to
#     * star_rating (default) : This takes the top two artists based on their average star-rating

# In[2]:


Find_The_Most_Popular_Music("most_listens")


# In[3]:


Find_The_Most_Popular_Music("most_users")


# In[4]:


Find_The_Most_Popular_Music("star_rating")


# In[5]:


Find_The_Most_Popular_Music(
    method="custom", 
    percentage_of_artists=10, 
    percentage_basis="star_rating",
    recommendation_basis="most_users")


# This custom recommendation query retrieved the top 10% of artists that received the highest average user rating, and returned Britney Spears and Depeche Mode as the artists who have the most users listening.

# # Guideline to use the Content-Based Recommender System

# There are two recommender systems created:
# 
# 1. A recommender system that calculates similarity based off only the top tag assigned to each artist.
# 
#     * This can be called using <u>content_based_recommendation_1(artists_name, artists_tags, tags)</u>
#     
#    
# 2. A recommender system that calculates similarity based off their name and the top five tags assigned to each artist.
# 
#     * This can be called using <u>content_based_recommendation_2(artists_name, artists_tags, tags)</u>
# 
# 
# <b>Note:</b> These functions may not run depending on system resources (Memory Error). If this is the case, you can check the notebook 'Content-Based Recommender System Workings' as there are examples shown there.

# In[6]:


artists_tags = pd.read_csv('../data/artists_tags.csv')
tags = pd.read_csv('../data/tags.csv')


# In[7]:


content_based_recommendation_1('Kanye West', artists_tags, tags)


# In[3]:


content_based_recommendation_2('Kanye West', artists_tags, tags)


# In[ ]:




