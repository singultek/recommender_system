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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


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
        self.tfidf = tfidf
        self.lsi = lsi
        self.reduced_space = reduced_space

        self.create_dataframes(dataset_path)
        self.dictionary_bag_of_words(movie_df=self.movies_df, tags_df=self.tags_df)
        self.movie_dictionary_matrix(tfidf=tfidf, lsi=lsi, reduced_space=reduced_space)

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
                                movie_df: pd.DataFrame,
                                tags_df: pd.DataFrame) -> dict:
        """
        The method to construct movie profiles with bag of words technique. In MovieLens case, genres and tags are used
        as features for creating movie profiles.
        Args:
            movie_df: The pandas dataframe with consisting of the movies and genres
            tags_df: The pandas dataframe with consisting of the movies and tags
        Return:
            self.movies_dictionary: The dictionary which indicates the dictionary(corpus) computed by bag of words technique
        """
        # Creating an empty dictionary to store key(movie_id) and values(genres and tags)
        self.movies_dictionary = {}

        # Constructing the key-value pairs for genres
        for _, row in movie_df.iterrows():
            if int(row['movieId']) not in self.movies_dictionary:
                self.movies_dictionary[int(row['movieId'])] = ''
            for genre in row['genres'].split('|'):
                self.movies_dictionary[int(row['movieId'])] += ' ' + genre.lower()

        # Constructing the key-value pairs for genres
        for _, row in tags_df.iterrows():
            self.movies_dictionary[int(row['movieId'])] += ' ' + row['tag'].lower()
        return self.movies_dictionary

    def movie_dictionary_matrix(self,
                                tfidf: bool,
                                lsi: bool,
                                reduced_space: int) -> (pd.DataFrame, int):
        """
        Creates the movies-dictionary(corpus) matrix.
        In the case of the MovieLens dataset, the dimension of matrix is n x m, where
        n is the number of movies and m the number of total words in the dictionary(corpus)
        created from the dataset.
        In case of lsi=True, dictionary reduction is applied with the SVD truncated
        at reduced_space and returns a reduction movies-dictiorany pd.dataframe.
        Args:
            tfidf: the boolean value which indicates whether TF-IDF technique will be applied to the counting or not
            lsi: the boolean value which indicates whether dictionary reduction with LSI technique will be applied or not
            reduced_space: the integer value which indicates the number of components of the reduced space when dictionary reduction is applied(lsi=True)
        Return:
            self.movies_dictionary_df, self.movies_dictionary_df.shape[1]: the tuple which contains
            movie-dictionary dataframe and the total number of words in dictionary
        """
        # Check the selected mode which indicates whether TF-IDF will be applied or not
        if tfidf:
            word_vectorizer = TfidfVectorizer()
        else:
            word_vectorizer = CountVectorizer()

        # Creating the pd.dataframe by using TF-IDF or Regular approach
        dense_matrix = word_vectorizer.fit_transform(self.movies_dictionary.values()).todense()
        column_feature_names = word_vectorizer.get_feature_names()
        movie_id_index = list(self.movies_dictionary.keys())
        self.movies_dictionary_df = pd.DataFrame(dense_matrix,
                                                 index=movie_id_index,
                                                 columns=column_feature_names)
        if lsi:
            # Convert the pd.dataframe into numpy matrix
            self.movies_dictionary_matrix = self.movies_dictionary_df.to_numpy()

            # Apply SVD for dictionary reduction
            svd = TruncatedSVD(n_components=reduced_space)
            self.reduction_movies_dictionary_df = svd.fit_transform(self.movies_dictionary_matrix)

            # Convert the dimension recuded numpy matrix into pd.dataframe
            self.reduction_movies_dictionary_df = pd.DataFrame(self.reduction_movies_dictionary_df,
                                                               index=movie_id_index)
        return self.movies_dictionary_df, self.movies_dictionary_df.shape[1]

    def movie_vector(self,
                     movie_id: int) -> np.array:
        """
        The method that returns the vector model of movie given by movieID
        Args:
            movie_id: the integer value that contains the OD of movie
        Return:
            self.movies_dictionary_df.loc[movie_id].to_numpy(): the np.array that contains
            the vector model of given movie_id
        """
        return self.movies_dictionary_df.loc[movie_id].to_numpy()


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

