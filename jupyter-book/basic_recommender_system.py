import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def Find_The_Most_Popular_Music(method="most_users", percentage_of_artists=90, percentage_basis="most_users", recommendation_basis="star_rating"):
    
    if method == "most_users":
        Find_The_Most_Popular_Music_By_Most_Users()
    elif method == "most_listens":
        Find_The_Most_Popular_Music_By_Most_Listens()
    elif method == "star_rating":
        Find_The_Most_Popular_Music_By_Star_Rating()
    elif method == "custom":
        if percentage_of_artists <= 0 or percentage_of_artists >= 100:
            raise Exception("Percentage of artists must be a number between but not including 0 and 100")
        elif percentage_basis not in ['most_users', 'most_listens', 'star_rating']:
            raise Exception("The percentage basis has to be one of: \n - 'most_users'\n - 'most_listens'\n - 'star_rating'")
        elif recommendation_basis not in ['most_users', 'most_listens', 'star_rating']:
            raise Exception("The recommendation basis has to be one of: \n - 'most_users'\n - 'most_listens'\n - 'star_rating'")
        else:
            Find_The_Most_Popular_Music_Custom(percentage_of_artists, percentage_basis, recommendation_basis)
    else:
        raise Exception("The method has to be one of: \n - 'most_users'\n - 'most_listens'\n - 'star_rating'\n - 'custom'")


def Find_The_Most_Popular_Music_By_Star_Rating():

    ## Importing the Necessary Datasets
    artists = pd.read_table('../data/artists.dat', sep="\t")
    user_artists_ratings = pd.read_csv('../data/user_artists_ratings.csv')
    
    ## Removing duplicate ID column
    user_artists_ratings.drop('Unnamed: 0', axis=1, inplace=True)
    
    ## Discovering the top 2 recommendations
    artists_highest_ratings = user_artists_ratings.sort_values(by='rating', ascending=False)
    artist1_rec = artists_highest_ratings.iloc[0,]['artistID']
    artist2_rec = artists_highest_ratings.iloc[1,]['artistID']
    
    ## Assigning the first and second artist details
    artist1_details = artists[artists['id']==artist1_rec]
    artist1_name = artist1_details.values[0][1]
    artist1_link = artist1_details.values[0][2]
    artist2_details = artists[artists['id']==artist2_rec]
    artist2_name = artist2_details.values[0][1]
    artist2_link = artist2_details.values[0][2]
    
     ## Print results
    print("We think you might like " + artist1_name + ", and you can view their profile through this link: " + artist1_link)
    print("You might also like " + artist2_name + ", and you can view their profile through this link: " + artist2_link)


def Find_The_Most_Popular_Music_By_Most_Users():
    ## Importing the Necessary Datasets
    artists = pd.read_table('../data/artists.dat', sep="\t")
    user_artists_ratings = pd.read_csv('../data/user_artists_ratings.csv')
    
    ## Removing duplicate ID column
    user_artists_ratings.drop('Unnamed: 0', axis=1, inplace=True)
    
    ## Assigning a new dataframe to include number of users, average rating and number of listens
    artists_user_count = (
    user_artists_ratings
    .groupby('artistID', as_index=False)
    .agg({'userID': ['count']}))
    
    ## Adjusting the column values
    artists_user_count.columns = ['artistID', 'count_users']
    
    ## Discovering the top 2 recommendations
    artists_most_users = artists_user_count.sort_values(by='count_users', ascending=False)
    artist1_rec = artists_most_users.iloc[0,]['artistID']
    artist2_rec = artists_most_users.iloc[1,]['artistID']
    
    ## Assigning the first and second artist details
    artist1_details = artists[artists['id']==artist1_rec]
    artist1_name = artist1_details.values[0][1]
    artist1_link = artist1_details.values[0][2]
    artist2_details = artists[artists['id']==artist2_rec]
    artist2_name = artist2_details.values[0][1]
    artist2_link = artist2_details.values[0][2]
    
     ## Print results
    print("We think you might like " + artist1_name + ", and you can view their profile through this link: " + artist1_link)
    print("You might also like " + artist2_name + ", and you can view their profile through this link: " + artist2_link)


def Find_The_Most_Popular_Music_By_Most_Listens():
    ## Importing the Necessary Datasets
    artists = pd.read_table('../data/artists.dat', sep="\t")
    user_artists_ratings = pd.read_csv('../data/user_artists_ratings.csv')
    
    ## Removing duplicate ID column
    user_artists_ratings.drop('Unnamed: 0', axis=1, inplace=True)
    
    ## Assigning a new dataframe to include number of users, average rating and number of listens
    artists_listen_count = (
    user_artists_ratings
    .groupby('artistID', as_index=False)
    .agg({'weight': ['sum']}))
    
    ## Adjusting the column values
    artists_listen_count.columns = ['artistID', 'count_listens']
    
    ## Discovering the top 2 recommendations
    artists_most_listens = artists_listen_count.sort_values(by='count_listens', ascending=False)
    artist1_rec = artists_most_listens.iloc[0,]['artistID']
    artist2_rec = artists_most_listens.iloc[1,]['artistID']
    
    ## Assigning the first and second artist details
    artist1_details = artists[artists['id']==artist1_rec]
    artist1_name = artist1_details.values[0][1]
    artist1_link = artist1_details.values[0][2]
    artist2_details = artists[artists['id']==artist2_rec]
    artist2_name = artist2_details.values[0][1]
    artist2_link = artist2_details.values[0][2]
    
     ## Print results
    print("We think you might like " + artist1_name + ", and you can view their profile through this link: " + artist1_link)
    print("You might also like " + artist2_name + ", and you can view their profile through this link: " + artist2_link)


def Find_The_Most_Popular_Music_By_Star_rating(percentage=10):
    
    ## Importing the Necessary Datasets
    artists = pd.read_table('../data/artists.dat', sep="\t")
    user_artists_ratings = pd.read_csv('../data/user_artists_ratings.csv')
    
    ## Removing duplicate ID column
    user_artists_ratings.drop('Unnamed: 0', axis=1, inplace=True)
    
    ## Assigning a new dataframe to include number of users, average rating and number of listens
    artists_total_listens = (
    user_artists_ratings
    .groupby('artistID', as_index=False)
    .agg({'userID': ['count'], 'rating': ['mean'], 'weight': ['sum']}))
    
    ## Adjusting the column values
    artists_total_listens.columns = ['artistID', 'count_users', 'avg_rating', 'total_listens']
    
    ## Filtering artists to top 10%
    top10 = np.percentile(artists_total_listens['total_listens'], 100-percentage)
    artists_highest_total_listens = artists_total_listens[artists_total_listens['total_listens']>=top10]
    
    ## Filtering the dataset to include only artists who have 5 or more users listening to their music
    artists_highest_listens = artists_highest_total_listens[artists_highest_total_listens['count_users']>=5]
    
    ## Discovering the top 2 recommendations
    artists_ordered_popularity = artists_highest_listens.sort_values(by='avg_rating', ascending=False)
    artist1_rec = artists_ordered_popularity.iloc[0,]['artistID']
    artist2_rec = artists_ordered_popularity.iloc[1,]['artistID']
    
    ## Assigning the first and second artist details
    artist1_details = artists[artists['id']==artist1_rec]
    artist1_name = artist1_details.values[0][1]
    artist1_link = artist1_details.values[0][2]
    artist2_details = artists[artists['id']==artist2_rec]
    artist2_name = artist2_details.values[0][1]
    artist2_link = artist2_details.values[0][2]
    
    ## Print results
    print("We think you might like " + artist1_name + ", and you can view their profile through this link: " + artist1_link)
    print("You might also like " + artist2_name + ", and you can view their profile through this link: " + artist2_link)


def Find_The_Most_Popular_Music_Custom(percentage_of_artists, percentage_basis, recommendation_basis):
    
    ## Importing the Necessary Datasets
    artists = pd.read_table('../data/artists.dat', sep="\t")
    user_artists_ratings = pd.read_csv('../data/user_artists_ratings.csv')
    
    ## Removing duplicate ID column
    user_artists_ratings.drop('Unnamed: 0', axis=1, inplace=True)
    
    ## Assigning a new dataframe to include number of users, average rating and number of listens
    artists_total_listens = (
    user_artists_ratings
    .groupby('artistID', as_index=False)
    .agg({'userID': ['count'], 'rating': ['mean'], 'weight': ['sum']}))
    
    ## Adjusting the column values
    artists_total_listens.columns = ['artistID', 'count_users', 'avg_rating', 'total_listens']
    
    ## Filtering artists to top percentage
    if percentage_basis == 'most_users':
        top_percentage = np.percentile(artists_total_listens['count_users'], 100-percentage_of_artists)
        top_artists = artists_total_listens[artists_total_listens['count_users']>=top_percentage]
    if percentage_basis == 'most_listens':
        top_percentage = np.percentile(artists_total_listens['total_listens'], 100-percentage_of_artists)
        top_artists = artists_total_listens[artists_total_listens['total_listens']>=top_percentage]
    if percentage_basis == 'star_rating':
        top_percentage = np.percentile(artists_total_listens['avg_rating'], 100-percentage_of_artists)
        top_artists = artists_total_listens[artists_total_listens['avg_rating']>=top_percentage]

    ## Discovering the top 2 recommendations
    if recommendation_basis == 'most_users':
        artists_ordered = top_artists.sort_values(by='count_users', ascending=False)
    elif recommendation_basis == 'most_listens':
        artists_ordered = top_artists.sort_values(by='total_listens', ascending=False)
    elif recommendation_basis == 'star_rating':
        artists_ordered = top_artists.sort_values(by='avg_rating', ascending=False)
    artist1_rec = artists_ordered.iloc[0,]['artistID']
    artist2_rec = artists_ordered.iloc[1,]['artistID']
    
    ## Assigning the first and second artist details
    artist1_details = artists[artists['id']==artist1_rec]
    artist1_name = artist1_details.values[0][1]
    artist1_link = artist1_details.values[0][2]
    artist2_details = artists[artists['id']==artist2_rec]
    artist2_name = artist2_details.values[0][1]
    artist2_link = artist2_details.values[0][2]
    
    ## Print results
    print("We think you might like " + artist1_name + ", and you can view their profile through this link: " + artist1_link)
    print("You might also like " + artist2_name + ", and you can view their profile through this link: " + artist2_link)
