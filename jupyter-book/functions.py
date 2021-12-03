import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_weights_mean_and_std(dataframe, userID=''):
    
    if userID:
        data = dataframe[dataframe['userID']==userID]['weight']
        title = "Weight Distribution of User #" + str(userID) + " with Mean and Standard Deviation"
    else:
        data = dataframe['weight']
        title = "Weight Distribution with Mean and Standard Deviation"
    plt.hist(data, weights=np.ones(len(data)) / len(data), bins=100)

    ## Assigning the mean and the standard deviation and then plotting them
    mean = data.mean()
    std = np.std(data)
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=1)
    plt.axvline(mean - (2.5 * std), color='k', linestyle='dashed', linewidth=1)
    plt.axvline(mean - (1.5 * std), color='k', linestyle='dashed', linewidth=1)
    plt.axvline(mean - (0.5 * std), color='k', linestyle='dashed', linewidth=1)
    plt.axvline(mean + (0.5 * std), color='k', linestyle='dashed', linewidth=1)
    plt.axvline(mean + (1.5 * std), color='k', linestyle='dashed', linewidth=1)
    plt.axvline(mean + (2.5 * std), color='k', linestyle='dashed', linewidth=1)

    plt.gca().set(
        title=title, 
        xlabel='Weight Values',
        ylabel='Frequency'
);


def remove_users_outliers(dataframe, userID):
    
    ## This method removes any outliers that are over 2 standard deviations away from the mean
    
    user_data = dataframe[dataframe['userID']==userID]
    user_mean = user_data['weight'].mean()
    user_std = np.std(user_data['weight'])
    
    new_user_data = user_data[abs(user_data['weight'] - user_mean) < (2 * user_std)]
    return new_user_data


def find_means_and_stds_per_user(dataframe):
    
    means = {}
    stds = {}
    
    for userID in dataframe.userID.unique():
        user_data = dataframe[dataframe['userID']==userID]
        user_mean = user_data['weight'].mean()
        user_std = np.std(user_data['weight'])
        means[userID] = user_mean
        stds[userID] = user_std
    
    return [means, stds]


# +
def find_mean(row, means):
    return means[row['userID']]

def find_std(row, stds):
    return stds[row['userID']]

def find_rating(row):
    weight = row['weight']
    user_mean = row['mean']
    user_std = row['std']
    if weight <= (user_mean - (1.5 * user_std)):
        return 1
    elif weight <= (user_mean - (0.5 * user_std)):
        return 2
    elif weight <= (user_mean + (0.5 * user_std)):
        return 3
    elif weight <= (user_mean + (1.5 * user_std)):
        return 4
    elif weight > (user_mean + (1.5 * user_std)):
        return 5


# -

def remove_all_outliers(dataframe):
    
    new_dataframe = dataframe[abs(dataframe['weight'] - dataframe['mean']) < (2 * dataframe['std'])]
    
    new_dataframe.drop(['std', 'rating', 'mean'], axis=1, inplace=True)
    
    new_dataframe['rating'] = 0
    new_dataframe['mean'] = 0
    new_dataframe['std'] = 0
    
    means_and_stds = find_means_and_stds_per_user(new_dataframe)
    means = means_and_stds[0]
    stds = means_and_stds[1]
    
    new_dataframe['mean'] = new_dataframe.apply(lambda row: find_mean(row, means), axis=1)
    new_dataframe['std'] = new_dataframe.apply(lambda row: find_std(row, stds), axis=1)
    new_dataframe['rating'] = new_dataframe.apply(lambda row: find_rating(row), axis=1)
    return new_dataframe
