# recommender_system
The project about building concrete recommender system by using Content-Based and Collaborative Filtering approaches

---
**Table of Contents**

* [Brief Introduction](#brief-introduction)
* [Directory structure](#directory-structure)
* [Explaination of project](#explaination-of-project)
* [Usage examples](#usage-examples)
* [Discussion of Results](#discussion-of-results)
* [About Author](#about-author)

---
## Brief Introduction

In this project, 2 main approaches of Recommender System will be presented. First one is the content based recommender approach. In this implementation, additional information about the user and/or item is explored. In this case, the genres of the movies is examined to find out recommendations for given userID. In the second approach, collaborative filtering based recommender system is applied. This time, recommender relies on the user-item interactions by using models such as KNN and SVD. Unfortunately, memory based techniques are not performed due to memory issues.
The dataset is can be found on the official [MovieLens](https://grouplens.org/datasets/movielens/) website. The small,100k version of the dataset is used for this project, that can be found in the [data](https://github.com/singultek/recommender_system/blob/main/data). As the official website implies, there are approxiamately:
* 100,000 ratings 
* 3,600 tag applications  
* 600 users

---
## Directory structure
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
## Explaination of project

The project can be divided into 2 main parts.

* Recommending by using Content Based Approach
* Recommending by using Collaborative Filtering Based Approach

[recommender_system](https://github.com/singultek/recommender_system/blob/main/recommender_system.py) main method should be used. For detailed usage options, please check the usage examples part. recommender_system imports [packages.utils]([recommender_system](https://github.com/singultek/recommender_system/blob/main/packages/utils.py)), thus it can parse the command line arguments and call the proper method from packages.utils (Ex. recommender_system.py can call packages.utils.content if the content based recommender system is selected).

---

### Recommending by using Content Based Approach

This part is responsible for getting the dataset, initializing the Content based approach and recommending movies. All the responsibilities of that part is computed with [packages.content](https://github.com/singultek/recommender_system/blob/main/packages/content.py). Inside this package, one can see the packages.content.ContentMovies(), packages.content.Users() and packages.content.Content() classes, which are built to store the movie, user information and recommend with respect to features of this information. 

The methods of packages.content.ContentMovies() are as following:

1. `__init__()`: 
  * 
  * 
2. `create_dataframes`: 
  * 
3. `dictionary_bag_of_words()`: 
  * 
  * 
  * 
4. `movie_dictionary_matrix()`: 
  *  
5. `movie_vector()`: 
  * 
  * 

The methods of packages.content.Users() are as following:

1. `__init__()`: 
  * 
  * 
2. `create_dataframes`: 
  * 
3. `__create_unique_lists()`: 
  * 
  * 
  * 
4. `users_movies_dict()`: 
  *  
5. `user_vector()`: 
  * 
  * 
6. `user_movie_summary()`: 
  * 
  *
The methods of packages.content.Content() are as following:

1. `__init__()`: 
  * 
  * 
2. `recommend()`: 
  *
  * 
  * 


---

### Recommending by using Collaborative Filtering Based Approach

This part is responsible for getting the dataset, initializing the Collaborative Filtering based approach and recommending movies. All the responsibilities of that part is computed with [packages.collab](https://github.com/singultek/recommender_system/blob/main/packages/collab.py). Inside this package, one can see the packages.content.CollabMovies() class, which is built to store the movie, user information and recommend with respect to features of this information. 

The methods of packages.collab.CollabMovies() are as following:

1. `__init__()`: 
  * 
  * 
2. `create_dataframes`: 
  * 
3. `__create_unique_lists()`: 
  * 
  * 
  * 
4. `choose_algorithm()`: 
  *  
5. `recommend()`: 
  * 
  * 


---
## Usage examples

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
One example of usage of train mode:

`$ python3 recommender_system.py content ./data 65 10`: builts a recommender system with content based approach with the dataset on `./data` for userID `65` and `10` maximum number of recommendations.

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

---

## Discussion of Results


                          
                               
---
## About Author

I'm Sinan Gültekin, a master student in Computer and Automation Engineering at the University of Siena. 

For any suggestions or questions, you can contact me via <singultek@gmail.com>

Distributed under the Apache-2.0 License. _See ``LICENSE`` for more information._
