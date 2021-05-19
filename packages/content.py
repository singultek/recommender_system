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
import pandas
import pandas as pd
import numpy as np


class Movies:
    def __init__(self,
                 dataset_path='./data',
                 tfidf=True,
                 lsi=True,
                 reduced_space=40) -> None:
        """
        Initializing the Movies class
        Args:
            dataset_path: the string with the path of dataset
                        By default, ./data folder path is given
            tfidf: the boolean value which indicates whether TF-IDF technique will be applied to the counting or not
                        By default, True is given
            lsi: the boolean value which indicates whether dictionary reduction with LSI technique will be applied or not
                        By default, True is given
            reduced_space: the integer value which indicates the number of components of the reduced space when dictionary reduction is applied(lsi=True)
                        By default, True is given
        Return:
            Nonw
        """
        self.dataset_path = dataset_path
        self.create_dataframes(dataset_path)
        self.dictionary_bag_of_words(movie_df=self.movies_df, tags_df=self.tags_df)
        print(self.__movies_dictionary)

    def create_dataframes(self,
                          dataset_path: str) -> None:
        """
        The method for reading the datasets and creating dataframe
        Args:
            dataset_path: the string with the path of dataset
        Return:
            None
        """
        # Creates movies dataframe
        movies_path = '{}/movies.csv'.format(dataset_path)
        self.movies_df = pd.read_csv(movies_path)

        # Creates tags dataframe
        tags_path = '{}/tags.csv'.format(dataset_path)
        self.tags_df = pd.read_csv(tags_path)
        self.tags_df = self.tags_df.drop('timestamp', axis=1)

    def dictionary_bag_of_words(self,
                                movie_df: pandas.DataFrame,
                                tags_df: pandas.DataFrame) -> None:
        """
        The method to construct movie profiles with bag of words technique. In MovieLens case, genres and tags are used
        as features for creating movie profiles.
        Args:
            movie_df: The pandas dataframe with consisting of the movies and genres
            tags_df: The pandas dataframe with consisting of the movies and tags
        Return:
            None
        """
        # Creating an empty dictionary to store key(movie_id) and values(genres and tags)
        self.__movies_dictionary = {}

        # Constructing the key-value pairs for genres
        for _, row in movie_df.iterrows():
            if int(row['movieId']) not in self.__movies_dictionary:
                self.__movies_dictionary[int(row['movieId'])] = ''
            for genre in row['genres'].split('|'):
                self.__movies_dictionary[int(row['movieId'])] += ' ' + genre.lower()

        # Constructing the key-value pairs for genres
        for _, row in tags_df.iterrows():
            self.__movies_dictionary[int(row['movieId'])] += ' ' + row['tag'].lower()

    def movie_dictionary_matrix(self):
        """

        :return:
        """

    def dictionary_reduction_latent_semantic_indexing(self):
        """

        :return:
        """


class Users:

    def __init__(self,
                 dataset_path=str,
                 movie_instance=object) -> None:
        """

        :param dataset_path:
        :param movie_instance:
        """


class Content:
    def __init__(self,
                 movie_instance=object,
                 user_instance=object):
        """

        :param movie_instance:
        :param user_instance:
        """

    def recommend(self, user_id, number_of_recommendation):
        """

        :param user_id:
        :param number_of_recommendation:
        :return:
        """
