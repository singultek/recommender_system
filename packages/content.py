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
            None
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
                 dataset_path: str,
                 movie_instance: Movies) -> None:
        """
        Initializing the Users class
        Args:
            dataset_path: the string with the path of dataset
                        By default, ./data folder path is given
            movie_instance: the object instance of Movies class
        Return:
            None
        """
        self.dataset_path = dataset_path
        self.create_dataframes(dataset_path)
        self.movies_list, self.users_list = self.__create_unique_lists(self.ratings_df)
        self.users_movies_dict(ratings_df=self.ratings_df)
        self.user_vector(movie_instance)

    def create_dataframes(self,
                          dataset_path: str) -> None:
        """
        The method for reading the datasets and creating dataframe
        Args:
            dataset_path: the string with the path of dataset
        Return:
            None
        """
        # Creates ratings dataframe
        ratings_path = '{}/ratings.csv'.format(dataset_path)
        self.ratings_df = pd.read_csv(ratings_path)

    @staticmethod
    def __create_unique_lists(dataframe: pd.DataFrame) -> (list, list):
        """
        The method for getting unique elements of dataframe
        Args:
            dataframe: the pd.Dataframe that will be computed to get unique elements
        Return:
            dataframe['movieId'].unique(), dataframe['userId'].unique(): the tuple of
            lists those contain the unique movie and user elements
        """
        return dataframe['movieId'].unique(), dataframe['userId'].unique()

    def users_movies_dict(self,
                          ratings_df: pd.DataFrame) -> dict:
        """
        The method constructs a dictionary with users-movies that rated by users as key-values pairs
        Args:
            ratings_df: the pd.Dataframe that contains the users and rated movies by those users
        Return:
            self.users_rated_movie_dictionary: the dictionary contains the movieId's of users' each rate
        """
        # Creating an empty dictionary to store key(users) and values(rated movies)
        self.users_rated_movie_dictionary = {}

        # Constructing the key-value pairs
        for user in self.users_list:
            self.users_rated_movie_dictionary[user] = {}
            # All ratings are product of a 0.5 between 0-5 range
            for rating in np.arange(0, 5.5, 0.5):
                # Adding movieId into correct user and rating match-ups
                self.users_rated_movie_dictionary[user][rating] = list(
                    ratings_df[(ratings_df['userId'] == user) & (ratings_df['rating'] == rating)]['movieId'])
        return self.users_rated_movie_dictionary

    def user_vector(self,
                    movie_instance: Movies) -> None:
        """
        The method that returns the vector model of users given by Movies object
        Args:
            movie_instance: the object instance of Movies class
        Return:
            None
        """
        # Creating an empty dictionary to store key(users) and values(rated movies)
        self.user_vectors = {}
        for user in self.users_list:
            # Numerator is the sum of all the movie-vectors multiplied by the rating
            numerator = 0
            # Denominator is the sum of all the ratings
            denominator = 0
            for rating in self.users_rated_movie_dictionary[user]:
                for movie in self.users_rated_movie_dictionary[user][rating]:
                    numerator += rating * movie_instance.movie_vector(movie)
                    denominator += rating
            # Weighted average
            self.user_vectors[user] = numerator / denominator

    def user_movie_summary(self,
                           user_id: int) -> (dict, list, dict):
        """
        The method that computes the dictionary describing the movies for each users,
        the non-rated movies of user_id and the vector model of the user given by user_id
        Args:
            user_id: ID of the user
        Return:
            self.users_rated_movie_dictionary[user_id], non_rated_list, self.user_vectors[user_id]:
            the dictionary describing the movies for each users,
            the non-rated movies of user_id,
            the vector model of the user given by user_id
        """
        # List of the movies those are rated by user_id
        rated_list = list(self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'])

        # List of all the movies
        all_movie_list = self.movies_list

        # List of the movies those are not rated by user_id
        non_rated_list = list(set(all_movie_list) - set(rated_list))
        return self.users_rated_movie_dictionary[user_id], non_rated_list, self.user_vectors[user_id]


class Content:
    def __init__(self,
                 movie_instance: Movies,
                 user_instance: Users,
                 user_id: int,
                 number_of_recommendation: int) -> None:
        """
        Initializing the Content class to compute Content-Based RecSys
        Args:
            movie_instance: the object instance of Movies class
            user_instance: the object instance of Users class
            user_id: ID of the user
            number_of_recommendation: the integer value indicates the total number of recommended movies
        Return:
            None
        """
        self.movies = movie_instance
        self.users = user_instance
        self.user_id = user_id
        self.number_of_recommendation = number_of_recommendation

    def recommend(self):
        # Initialize the recommendations
        recommendations = [[0, 0]] * self.number_of_recommendation
        _, non_rated_movies_list, user_vector = self.users.user_movie_summary(self.user_id)
        user_vector = user_vector.reshape(1, -1)

        for movieID in non_rated_movies_list:
            movie_vector = self.movies.movie_vector(movieID).reshape(1, -1)
            # Compute the similarity of user vector and movie vector with cosine similarity measure
            similarity = float(cosine_similarity(user_vector, movie_vector))

            # Since the recommendations are sorted in ascending order, one can confidently insert the similarity in the list when similarity > recommendations[0][0]
            if similarity > recommendations[0][0]:
                recommendations[0] = [similarity, movieID]
                recommendations = sorted(recommendations, key=lambda x: x[0])

        # Append movie title and genre into recommendation list and convert more readable pd.Dataframe data type
        for i in range(int(self.number_of_recommendation)):

            similarity_measure = round(recommendations[i][0], 3)
            print(similarity_measure)
            print(self.movies.movies_df[self.movies.movies_df['movieId'] == recommendations[i][1]].values)
            recommended_movie_id = self.movies.movies_df[self.movies.movies_df['movieId'] == recommendations[i][1]].values[0][0]
            recommended_movie_title = self.movies.movies_df[self.movies.movies_df['movieId'] == recommendations[i][1]].values[0][1]
            recommended_movie_genre = self.movies.movies_df[self.movies.movies_df['movieId'] == recommendations[i][1]].values[0][2]

            recommendations[i][0] = similarity_measure
            recommendations[i][1] = recommended_movie_id
            recommendations[i].append(recommended_movie_title)
            recommendations[i].append(recommended_movie_genre)

        recommendations = pd.DataFrame(recommendations, columns=['similarity', 'movieId', 'title', 'genres'])
        recommendations.sort_values(by=['similarity'], ascending=False, inplace=True, ignore_index=True)

        return recommendations
