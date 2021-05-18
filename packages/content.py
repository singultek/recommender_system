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
import pandas as pd
import numpy as np


class Movies:
    def __init__(self,
                 dataset_path=str,
                 tfidf=bool,
                 lsi=bool,
                 reduced_space=int) -> None:
        """
        Initializing the Movies class
        Args:
            dataset_path: the string with the path of dataset
            tfidf: the boolean value which indicates whether TF-IDF technique will be applied to the counting or not
            lsi: the boolean value which indicates whether dictionary reduction with LSI technique will be applied or not
            reduced_space: the integer value which indicates the number of components of the reduced space when dictionary reduction is applied(lsi=True)
        Return:
            Nonw
        """
        self.dataset_path = dataset_path
        self.create_dataframes(dataset_path)

    def create_dataframes(self, dataset_path):
        """

        :param dataset_path:
        :return:
        """
        # Creates movies dataframe
        movies_path = '{}/movies.csv'.format(dataset_path)
        self.movies_df = pd.read_csv(movies_path)

        # Creates tags dataframe
        tags_path = '{}/tags.csv'.format(dataset_path)
        self.tags_df = pd.read_csv(tags_path)
        self.tags_df = self.tags_df.drop('timestamp', axis=1)

    def dictionary_bag_of_words(self):
        """

        :return:
        """

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
