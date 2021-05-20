"""
Copyright 2021 - Sinan Gültekin <singultek@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Import the necessary libraries
import argparse

from .content import *
from .collaborative import *


def parse_arguments() -> argparse.Namespace:
    """
    The method is created for enhancing user interface on the command line by parsing command line arguments.
    Args:
    Returns:
        The Namespace object which holds the input arguments given from command line
    """

    # Creating subparsers for each working mode
    parser = argparse.ArgumentParser(description='The command line argument parser for Recommender System')
    subparsers_set = parser.add_subparsers(title='Working Approach of Recommender System',
                                           description='Main 2 approaches of selecting which Recommender System will work',
                                           dest='approach',
                                           required=True,
                                           help='Decide the working mode from following options: '
                                                'content = Content-Based Recommender System, '
                                                'collab = Collaborative Filtering Recommender System')

    content_parser = subparsers_set.add_parser('content',
                                               description='Content-Based Approach',
                                               help='RecSys with using Content-Based Approach')
    collab_parser = subparsers_set.add_parser('collab',
                                              description='Collaborative Filtering Approach',
                                              help='RecSys with using Collaborative Filtering Approach')

    # Adding subparsers for content approaches
    content_parser.add_argument('dataset_path',
                                type=str,
                                help='A dataset folder where the data is stored')
    content_parser.add_argument('userID',
                                type=int,
                                help='A integer value which indicates the userID to recommend movies')
    content_parser.add_argument('num_recommendation',
                                type=int,
                                help='A integer value which indicates the number of recommention given for userID')
    content_parser.add_argument('--tfidf',
                                default=True,
                                type=bool,
                                help='(default = True) The boolean value which indicates whether TF-IDF technique will be applied to the counting or not')
    content_parser.add_argument('--lsi',
                                default=True,
                                type=bool,
                                help='(default = True) The boolean value which indicates whether dictionary reduction with LSI technique will be applied or not')

    # Adding subparsers for collab approach
    collab_parser.add_argument('dataset_path',
                               type=str,
                               help='A dataset folder where the data is stored')
    collab_parser.add_argument('userID',
                               type=int,
                               help='A integer value which indicates the userID to recommend movies')
    collab_parser.add_argument('num_recommendation',
                               type=int,
                               help='A integer value which indicates the number of recommention given for userID')
    collab_parser.add_argument('--algorithm',
                               # default='UserUser',
                               type=str,
                               help='(default = UserUser) A Collaborative Filtering RecSys Approach which will be used to recommend')

    args_parsed = parser.parse_args()
    return args_parsed


def content(dataset_path: str,
            user_id: int,
            num_recommendation: int,
            tfidf: bool,
            lsi: bool) -> None:
    """
    The main content method to perform the recommending the movie with content-based approach
    Args:
        dataset_path: the string with the path of dataset
        user_id: A integer value which indicates the userID to recommend movies
        num_recommendation: A integer value which indicates the number of recommention given for userID
        tfidf: the boolean value which indicates whether TF-IDF technique will be applied to the counting or not
        lsi: the boolean value which indicates whether dictionary reduction with LSI technique will be applied or not
    Returns:
        Content: the Content object that computes and gives the recommendations
    """
    # Initialize the movie object
    movies = ContentMovies(dataset_path=dataset_path, tfidf=tfidf, lsi=lsi, reduced_space=40)
    users = Users(dataset_path, movies)

    # Check the given inputs whether they are in the range or not
    if user_id not in users.users_list:
        raise ValueError('Please enter a valid user id in the range of 1-{}'.format(len(users.users_list)))
    if num_recommendation > len(users.movies_list):
        raise ValueError(
            'Please enter a valid number of recommendation in the range of 1-{}'.format(len(users.movies_list)))

    # Initiate the Content instance to get recommendations
    content_rec_sys = Content(movies,
                              users,
                              user_id=user_id,
                              number_of_recommendation=num_recommendation)

    # Print the recommended movies list for given user_id and number_of_recommendation
    print(content_rec_sys.recommend())


def collab(dataset_path: str,
           user_id: int,
           num_recommendation: int,
           algorithm: str) -> None:
    """
    The main collab method to perform the recommending the movie with collaborative filtering approach
    Args:
        dataset_path: the string with the path of dataset
        user_id: A integer value which indicates the userID to recommend movies
        num_recommendation: A integer value which indicates the number of recommention given for userID
        algorithm: the string with the name of algorithm to be used
    Returns:
        None
    """
    # Initialize the movie object
    movies = CollabMovies(dataset_path=dataset_path, algorithm=algorithm)

    # Check the given inputs whether they are in the range or not
    if user_id not in movies.users_list:
        raise ValueError('Please enter a valid user id in the range of 1-{}'.format(len(movies.users_list)))
    if num_recommendation > len(movies.movies_list):
        raise ValueError(
            'Please enter a valid number of recommendation in the range of 1-{}'.format(len(movies.movies_list)))

    #print(user_id)
    #print(num_recommendation)
    #print(algorithm)
