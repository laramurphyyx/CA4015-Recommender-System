# Background

## What are Recommender Systems?

Recommender systems are machine-learning based algorithms that can 'learn' what a user might like to see, based on the information it has about the user or the product/item. Recommender systems can be seen in many services that we use regularly, such as Amazon recommending products to us or Facebook recommending people we may know. Some companies rely heavily on the use of these techniques and algorithms, such as Netflix, where their business model and its success are influenced by the accuracy of their recommendations.

## How do Recommender Systems Work?

There are several approaches that can be taken to build a recommender system. Some of which are:

* <b>Simple Recommenders</b>: These are the easiest to implement. This is a simple recommender system that shows the user what is popular. This is a generalised method of recommending, although the idea behind this system is that things that are more popular and more liked will be more likely to be enjoyed by the average audience.

* <b>Collaborative Filtering</b>: This method is more complex than the above Simple Recommender, as it is more specific to the user. This method ignores information about the target item (song, movie, etc.), and just bases its recommendation purely on user information/opinions. The concept of collaborative filtering is based on the idea that user A has the same likes/dislikes/opinions as user B, that user A is more likely to have the same opinion as user B on a different issue than a randomly chosen user. ([Interesting link about user-based vs. item-based](https://www.datacamp.com/community/tutorials/recommender-systems-python))

* <b>Content-Based Recommenders</b>: This approach uses the item's metadata, for example the metadata held for a movie would be it's genre, director, description or actors. This algorithm identifies the genres/directors/actors that a user generally likes, and uses this as a guide for future recommendations.

* <b>Hybrid Approach</b>: This approach is a combination of collaborative filtering and content-based recommenders. This can be quite difficult to implement, although when optimized, can outperform both methods of recommending.

## About the Data used in our Recommender System

Our Recommender System is based on artists from the LastFM dataset.
The LastFM dataset contains information about 92,800 artists and 1,892 users. The data is comprised of 6 files:

- artists.dat : This file contains information about the artist, such as their ID, their name, and URLs associated with them
- tags.dat : This file contains a mapping for each tag ID and the tag description
- user_artists.dat : This file contains information about how many times a user has listened to each artist
- user_friends.dat : This file contains a mapping for a user and each of their friends
- user_taggedartists.dat : This file contains information about the tags assigned to each artist from each user
- user_taggedartists_timestamps.dat : This file contains the same content as user_taggedartists.dat, but with the added timestamp

The dataset also includes a readme file to provide context and further explanation on the information contained in each file. 

This dataset is available through [this link](https://grouplens.org/datasets/hetrec-2011/), or you can download it directly by clicking [here](http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip).
