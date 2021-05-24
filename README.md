# recommender_system
The project about building concrete recommender system by using Content-Based and Collaborative Filtering approaches

---
**Table of Contents**

* [Brief Introduction](#brief-introduction)
* [Directory Structure](#directory-structure)
* [Explaination of Project](#explaination-of-project)
* [Usage Examples](#usage-examples)
* [Discussion of Results](#discussion-of-results)
* [About Author](#about-author)

---
## Brief Introduction

In this project, 2 main approaches of Recommender System will be presented. First one is the content based recommender approach. In this implementation, additional information about the user and/or item is explored. In this case, the genres and tags of the movies is examined to find out recommendations for given userID. In the second approach, collaborative filtering model based recommender system is applied. This time, recommender relies on the user-item interactions by using [K-Nearest Neighbour(KNN)](https://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBaseline) clustering model and [Singular Value Decomposition(SVD)](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD) matrix factorization model. Unfortunately, collaborative filtering memory based techniques are not performed due to memory issues.
The dataset is can be found on the official [MovieLens](https://grouplens.org/datasets/movielens/) website. The small,100k version of the dataset is used for this project, that can be found in the [data](https://github.com/singultek/recommender_system/blob/main/data). As the official website implies, there are approxiamately:
* 100,000 ratings
* 9000 movies
* 3,600 tag applications  
* 600 users

---
## Directory Structure
```
.
├── data
│   ├── movies.csv
│   ├── ratings.csv
│   ├── tags.csv
├── packages
│   ├── __init__.py
│   ├── collaborative.py
│   ├── content.py
│   └── utils.py
├── LICENSE
├── README.md
├── Terminal Command Run Examples.txt
├── __init__.py
├── recommender_system.py
├── requirements.txt
```

---
## Explaination of Project

The project can be divided into 2 main parts.

* Recommending by using Content Based Approach
* Recommending by using Collaborative Filtering Based Approach

[recommender_system](https://github.com/singultek/recommender_system/blob/main/recommender_system.py) main method should be used. For detailed usage options, please check the usage examples part. recommender_system imports [packages.utils](https://github.com/singultek/recommender_system/blob/main/packages/utils.py), thus it can parse the command line arguments and call the proper method from packages.utils (Ex. recommender_system.py can call packages.utils.content if the content based recommender system is selected).

---

### Recommending by using Content Based Approach

This part is responsible for getting the dataset, initializing the Content based approach and recommending movies. All the responsibilities of that part is computed with [packages.content](https://github.com/singultek/recommender_system/blob/main/packages/content.py). Inside this package, one can see the packages.content.ContentMovies(), packages.content.Users() and packages.content.Content() classes, which are built to store the movie, user information and recommend with respect to features of this information. 

The methods of packages.content.ContentMovies() are as following:

1. `__init__()`: Class constructer
  * Checks the given data_path
  * Checks the tfidf boolean value which stands for term frequency-inverse document frequency and decided which word counter will be used between [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn-feature-extraction-text-tfidfvectorizer) and [regular word counter](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn-feature-extraction-text-countvectorizer) 
  * Checks the lsi boolean values which stands for latent semantic indexing and decided whether [dictionary dimension reduction(with SVD truncated technique)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn-decomposition-truncatedsvd) will be applied, or not
  * Calls the methods for further steps
2. `create_dataframes`: Creating pandas dataframes method
  * Gets the given data_path, reads and creates dataframes for movies and tags
3. `dictionary_bag_of_words()`: Construct movie profiles method
  * Creates a dictionary(corpus) for movie profiles by using genres and tags
  * Uses Bag of Word(BoW) technique for storing genres and tags for each movie
4. `movie_dictionary_matrix()`: Constructing movies-dictionary(corpus) matrix method
  * Creates the movies-dictionary(corpus) matrix, the dimension of matrix is n x m, where n is the number of movies and m the number of total words in the dictionary(corpus) created from the dataset.
  * Decides whether dictionary dimension reduction(with SVD truncated technique) will be applied(lsi=True), or not
5. `movie_vector()`: Getting movie vector method
  * Returns the vector model of movie given by movieID

The methods of packages.content.Users() are as following:

1. `__init__()`: Class constructer
  * Checks the given data_path
  * Calls the methods for further steps
2. `create_dataframes`: Creating pandas dataframes method
  * Gets the given data_path, reads and creates dataframes for ratings
3. `__create_unique_lists()`: Getting unique elements method
  * Returns to the list of unique elements of given input dataframes 
4. `users_movies_dict()`: Constructing user-movie dict method
  * Constructs a dictionary with users-movies that rated by users as key-values pairs
  * The dicitonary has user and rating keys and movies are stored as values between these nested keys
5. `user_vector()`: Getting user vector method
  * Gets the weighted average of ratings for movies for each user in order to understand the user's average rating trends for movies
6. `user_movie_summary()`: Getting summary of key points method
  * Computes the non rated list of the movies for each user 
  * Returns the vector of the user given by userID

The methods of packages.content.Content() are as following:

1. `__init__()`: Class constructer
  * Initialize the user and movie objects, those are constructed by the help of previous classes
2. `recommend()`: Predicting recommended movies method
  * Computing similarities of user and movie vectors with cosine similarity measure
  * Sorting the results by similarity and give the most similar recommendations


---

### Recommending by using Collaborative Filtering Based Approach

This part is responsible for getting the dataset, initializing the Collaborative Filtering based approach and recommending movies. All the responsibilities of that part is computed with [packages.collab](https://github.com/singultek/recommender_system/blob/main/packages/collab.py). Inside this package, one can see the packages.content.CollabMovies() class, which is built to store the movie, user information and recommend with respect to features of this information. 

The methods of packages.collab.CollabMovies() are as following:

1. `__init__()`: Class constructer
  * Checks the given data_path
  * Checks the given algorithm to be used
  * Calls the methods for further steps
2. `create_dataframes`: Creating pandas dataframes method
  * Gets the given data_path, reads and creates dataframes for movies and ratings
3. `__create_unique_lists()`: Getting unique elements method
  * Returns to the list of unique elements of given input dataframes 
4. `choose_algorithm()`: Selecting the given algorithm method
  * Gets the given algorithm input and creates a model from the given algorithm 
  * [K-Nearest Neighbour(KNN)](https://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBaseline) and [Singular Value Decomposition(SVD)](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD) algorithms are the options for collaborative filtering model based approach
  * KNN performs clustering and SVD performs matrix factorization techniques
5. `recommend()`: Predicting recommended movies method
  * Gets the userID, number of recommendations and algorithm as input
  * Creates [surprise.Reader](https://surprise.readthedocs.io/en/stable/reader.html#surprise.reader.Reader) and [surprise.Dataset](https://surprise.readthedocs.io/en/stable/dataset.html#surprise.dataset.Dataset.load_from_df) instances
  * Trains the model of given algorithm
  * Computes the accuracy with Root Mean Squared Error(RMSE)
  * Returns the recommended movies


---
## Usage Examples

The main [recommender_system](https://github.com/singultek/recommender_system/blob/main/recommender_system.py) code can run in two working modes: content based and collaborative filtering based approaches. Following help commands and usage examples can be followed to run the code from command line:


### Content

In order to see the help for content mode, one can run following on the command line:
`$ python3 recommender_system.py content -h`

```

usage: recommender_system.py content [-h] [--tfidf TFIDF] [--lsi LSI] dataset_path userID num_recommendation

Content-Based Approach

positional arguments:
  dataset_path        A dataset folder where the data is stored
  userID              A integer value which indicates the userID to recommend movies
  num_recommendation  A integer value which indicates the number of recommention given for userID

optional arguments:
  -h, --help          show this help message and exit
  --tfidf TFIDF       (default = True) The boolean value which indicates whether TF-IDF technique will be applied to the counting or not
  --lsi LSI           (default = True) The boolean value which indicates whether dictionary reduction with LSI technique will be applied or not

```
One example of usage of content based recommender system:

`$ python3 recommender_system.py content ./data 65 10`: builts a recommender system with content based approach with 
 * the dataset on `./data` 
 * for userID `65` 
 * for maximum number of recommendations `10`.

### Collaborative


In order to see the help for collaborative mode, one can run following on the command line:
`$ python3 recommender_system.py collab -h`

```
usage: recommender_system.py collab [-h] dataset_path userID num_recommendation algorithm

Collaborative Filtering Approach

positional arguments:
  dataset_path        A dataset folder where the data is stored
  userID              A integer value which indicates the userID to recommend movies
  num_recommendation  A integer value which indicates the number of recommention given for userID
  algorithm           A Collaborative Filtering RecSys Approach which will be used to recommend

optional arguments:
  -h, --help          show this help message and exit
```
One example of usage of collaborative filtering based recommender system:

`$ python3 recommender_system.py collab ./data 65 10 KNN-cosine-user`: builts a recommender system with collaborative filtering based approach with
 * the dataset on `./data`
 * for userID `65` 
 * for maximum number of recommendations `10`
 * implementing `KNN` algorithm with `user` based and using `cosine` similarity measures.


---

## Discussion of Results


| model                                              |    RMSE   | 
|----------------------------------------------------|-----------|
| **collab ./data 65 10 SVD-50-0.002**               | **0.486** |
| **collab ./data 65 10 SVD-50-0.001**               | **0.430** |
| **collab ./data 65 10 SVD-50-0.0001**              | **0.233** |
| collab ./data 65 10 KNN-cosine-user                |   0.824   |
| collab ./data 65 10 KNN-cosine-item                |   0.531   |
| collab ./data 65 10 KNN-pearson-user               |   0.795   |
| **collab ./data 65 10 KNN-pearson-item**           | **0.517** |
| collab ./data 65 10 KNN-msd-user                   |   0.826   |
| collab ./data 65 10 KNN-msd-item                   |   0.546   |
| collab ./data 65 10 KNN-pearson_baseline-user      |   0.761   |
| collab ./data 65 10 KNN-pearson_baseline-item      |   0.533   |

When the results are examined, 3 main outcome can be noticed immediately. First observation is that, matrix factorization method(SVD) predominates the results by far comparing to clustering method(KNN). The difference between the performance of methods increase dramatically by decreaseing the learning rate for SVD method. There can be applied grid search technique to identify the best parameters in the range of several parameters, but author didn't see that necessary for this case project. 

Related with the first outcome, one can state the second outcome as that, item based similarity measures compute better result than user based similarities. One reason for that is item set is more robust and stable comparing to user set, thus it gives slightly more concrete recommendations. Additionally, users may have multiple tastes and that makes hard to predict with user based method comparing to item based one. These reasons may be the indications of why item based method is more preferable for large scaled projects, for example Amazon uses item based method because the changes in the user set becomes inefficient and extremely dynamic for computing. On the contrary, Amazon's number of item set  consists of more or less same products.   

Third and the last outcome is that similarity measures that are used in that project don't affect the overall performance far too much. There are domination of pearson similarity measure slightly, but one cannot consider a little difference as concrete outcome for all collaborative filtering projects.  

---

## About Author

I'm Sinan Gültekin, a master student in Computer and Automation Engineering at the University of Siena. 

For any suggestions or questions, you can contact me via <singultek@gmail.com>

Distributed under the Apache-2.0 License. _See ``LICENSE`` for more information._

