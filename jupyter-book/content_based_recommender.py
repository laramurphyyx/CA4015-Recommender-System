import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tags_df = pd.read_csv('../data/tags.csv')
artist_tags_df = pd.read_csv('../data/artists_tags.csv')


def get_tags_dictionary(tags_df=tags_df):
    
    tag_dict = {}
    tag_dict[0] = ""
    for row in tags_df.itertuples():
        tagID = row[1]
        tag = row[2]
        tag_dict[tagID] = tag
        
    return tag_dict


def get_recommendations(name, cosine_sim):
    
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


def content_based_recommendation_1(artist_name, artist_tags_df=artist_tags_df, tags_df=tags_df):
    
    # Assigning the tags dictionary to swap the ID values for their string values in the artists_tags dataframe
    tag_dict = get_tags_dictionary(tags_df)
    artist_tags_df['first_tag'] = artist_tags_df.apply(lambda row: tag_dict[row['first_tag']], axis=1)
    artist_tags_df['second_tag'] = artist_tags_df.apply(lambda row: tag_dict[row['second_tag']], axis=1)
    
    # Calculating the TF-IDF score for each artist and token and storing it in a matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(artist_tags_df['first_tag'])
    
    # Defining the cosine similarity and storing in a matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Creating an inverted index for easier calling of artist names
    indices = pd.Series(artist_tags_df.index, index=artist_tags_df['name']).drop_duplicates()
    
    return get_recommendations(artist_name, cosine_sim)


def find_name_and_tags(dataframe):
    return "".join(dataframe['name'].split()) + ' ' + dataframe['top_five_tags']


def content_based_recommendation_2(artist_name, artist_tags_df=artist_tags_df, tags_df=tags_df):
    
    # Assigning the tags dictionary to swap the ID values for their string values in the artists_tags dataframe
    tag_dict = get_tags_dictionary(tags_df)
    artist_tags_df['first_tag'] = artist_tags_df.apply(lambda row: tag_dict[row['first_tag']], axis=1)
    artist_tags_df['second_tag'] = artist_tags_df.apply(lambda row: tag_dict[row['second_tag']], axis=1)
    artist_tags_df['top_five_tags'] = artist_tags_df['top_five_tags'].fillna('')
    artist_tags_df['name_and_tags'] = artist_tags_df.apply(find_name_and_tags, axis=1)
    
    # Using the CountVectoriser() instead of the TF-IDF
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(artist_tags_df['name_and_tags'])
    
    # Defining the cosine similarity and storing in a matrix
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    # Creating an inverted index for easier calling of artist names
    indices = pd.Series(artist_tags_df.index, index=artist_tags_df['name']).drop_duplicates()
    
    return get_recommendations(artist_name, cosine_sim)
