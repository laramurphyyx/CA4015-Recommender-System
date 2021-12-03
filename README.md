# CA4015-Recommender-System

## Assignment Details

This project is a joint assignment for the third and fourth assignment of CA4015, worth 50% of the module.

The assignment users recommender system algorithms using the Last.FM dataset to recommend artists.

The assignment follows this [tutorial](https://developers.google.com/machine-learning/recommendation) about Collaborative Filter Recommender Systems on the MovieLens dataset. 
In addition to that tutotial, I have also implemented:

* A Basic Recommender System: A recommender system that will recommend artists based on their ratings/number of listeners
* A Content-Based Recommender System: A recommender system that will recommend artists based on their tags/name
* A method to Assign Star-Ratings to Artist: Identifying a suitable way to assign star ratings for each artist from the users


## Repository Layout

* <b>README.md:</b> This markdown file, which outlines the assignment description and deliverables.

* <b>data:</b> The repository which contains a jupyter notebook to import the data directly from [this site](https://grouplens.org/datasets/hetrec-2011/), and cleans them slightly.

* <b>jupyter-book:</b> This repository contains all of the required files to build our jupyter notebook. This is where all of our code, notebooks and markdown files are kept.


## Deliverables

The deliverable for this assignment is the complete jupyter book.

The link to the interactive jupyter book is [here](https://laramurphyyx.github.io/CA4015-Recommender-System/Introduction.html).

Otherwise the book can be accessed in the form of a PDF [here](https://github.com/laramurphyyx/CA4015-Recommender-System/blob/main/jupyter-book/_build/pdf/book.pdf).

The github repository can be accessed through [this link](https://github.com/laramurphyyx/CA4015-Recommender-System).

## Encountered Bugs/Errors

The following errors should be noted when reading the jupyter book:

1. There is a bug where the last page disappears from the page navigation panel on the left of the jupyter book. The last page 'Using the Recommender Systems' appears when you have the 'Background' page open, although will disappear once you access another page.

3. There is a keyboard interrupt error that occasionally occurs when running the 'Content-Based Recommender System' or the 'Collaborative Filtering Recommender System', possibly due to a timeout. This can be seen in the PDF but does not occur in the notebook in the repository.

4. There is a memory issue when trying to run the content-based recommender system from a function, in 'Using the Recommender Systems'. This may work if ran locally on a different machine, but examples are given in the preceding notebook 'Content-Based Recommender'.
