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
import pandas as pd
import numpy as np

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
    collab_parser.add_argument('--algorithm',
                               default='UserUser',
                               type=str,
                               choices=['UserUser', 'ItemItem'],
                               help='(default = UserUser) A Collaborative Filtering RecSys Approach which will be used to recommend')

    args_parsed = parser.parse_args()
    return args_parsed


def content(dataset_path: str,
            tfidf: bool,
            lsi: bool) -> None:
    """
    The main content method to perform the recommending the movie with content-based approach
    Args:
        dataset_path: the string with the path of dataset
        tfidf: the boolean value which indicates whether TF-IDF technique will be applied to the counting or not
        lsi: the boolean value which indicates whether dictionary reduction with LSI technique will be applied or not
    Returns:
        None
    """
    # Read the datasets into pandas dataframe
    movies_path = '{}/movies.csv'.format(dataset_path)
    movies_df = pd.read_csv(movies_path)

    ratings_path = '{}/ratings.csv'.format(dataset_path)
    ratings_df = pd.read_csv(ratings_path)
    ratings_df = ratings_df.drop('timestamp', axis=1)

    tags_path = '{}/tags.csv'.format(dataset_path)
    tags_df = pd.read_csv(tags_path)
    tags_df = tags_df.drop('timestamp', axis=1)
    initialize_content(movies_df, ratings_df, tags_df)
    return


def collab(dataset_path: str,
           algorithm: str) -> None:
    """
    The main collab method to perform the recommending the movie with collaborative filtering approach
    Args:
        dataset_path: the string with the path of dataset
        algorithm: the string with the name of algorithm to be used
    Returns:
        None
    """
    # Read the datasets into pandas dataframe
    movies_path = '{}/movies.csv'.format(dataset_path)
    movies_df = pd.read_csv(movies_path)

    ratings_path = '{}/ratings.csv'.format(dataset_path)
    ratings_df = pd.read_csv(ratings_path)
    ratings_df = ratings_df.drop('timestamp', axis=1)

    # Initialize the Collaborative Filtering Approach
    initialize_collab(ratings_df)
    return
