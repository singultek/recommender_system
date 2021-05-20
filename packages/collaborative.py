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

from surprise import Reader
from surprise import Dataset
from surprise import KNNBaseline, SVD


class CollabMovies:
    def __init__(self,
                 dataset_path='./data',
                 algorithm='KNN-pearson-user') -> None:
        """
        Initializing the Movies class
        Args:
            dataset_path: the string with the path of dataset
                        By default, ./data folder path is given
        Return:
            None
        """
        self.dataset_path = dataset_path

        self.create_dataframes(dataset_path)
        self.movies_list, self.users_list = self.__create_unique_lists(self.ratings_df)
        self.choose_algorithm(algorithm=algorithm)
        return

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

        # Creates ratings dataframe
        ratings_path = '{}/ratings.csv'.format(dataset_path)
        self.ratings_df = pd.read_csv(ratings_path)
        self.ratings_df = self.ratings_df.drop('timestamp', axis=1)
        return

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

    def choose_algorithm(self,
                         algorithm: str) -> None:
        """
        The method that selects the algorithm from the command line user input
        Args:
            algorithm: the string value that stored the details of chosen algorithm
        Return:
            None
        """
        # Initialize the empty string for selected algorithm
        self.selected_algorithm = ''

        # Executing the selected algorithm with respect to value of string
        if algorithm is not None:
            # First string value indicates the backbone of approach
            backbone = algorithm.split('-')[0]
            # Check the possible options
            if backbone == 'KNN' and len(algorithm.split('-')) == 3:
                print('KNN model based collaborative approach is selected!')
                similarity_measure = algorithm.split('-')[1]
                user_based = algorithm.split('-')[2]
                if (similarity_measure == 'cosine' or similarity_measure == 'msd' or similarity_measure == 'pearson' or similarity_measure == 'pearson_baseline') and (user_based == 'user'):
                    self.selected_algorithm = KNNBaseline(sim_options={'name': similarity_measure,
                                                                       'user_based': True})
                elif (similarity_measure == 'cosine' or similarity_measure == 'msd' or similarity_measure == 'pearson' or similarity_measure == 'pearson_baseline') and (user_based == 'item'):
                    self.selected_algorithm = KNNBaseline(sim_options={'name': similarity_measure,
                                                                       'user_based': False})
                else:
                    raise ValueError('The given parameters {} for KNN algorithm is not recognized. Please check the propoer format'.format(str(backbone) + '-' + str(similarity_measure) + '-' + str(user_based)))
            elif backbone == 'SVD' and len(algorithm.split('-')) == 3:
                print('SVD model based collaborative approach is selected!')
                epoch = int(algorithm.split('-')[1])
                learning_rate = float(algorithm.split('-')[2])
                self.selected_algorithm = SVD(n_epochs=epoch, lr_all=learning_rate)
            elif backbone == '':
                raise ValueError('Please give an algorithm! Algorithm cannot be left empty')
            else:
                raise ValueError(
                    'The given algorithm format {} is not recognized. Please check the propoer format'.format(
                        str(backbone)))
        elif algorithm is None:
            # Raise an error message for None value for algorithm
            raise ValueError('Please give an algorithm! Algorithm cannot be left empty')
        return

    def recommend(self,
                  user_id: int,
                  number_of_recommendation: int,
                  selected_algorithm: KNNBaseline or SVD) -> pd.DataFrame:
        """
        The method that predicts the movie recommendations
        Args:
            user_id: the integer value that stored the userID that prediction is made for
            number_of_recommendation: the integer value that indicates the required number of recommendations
            selected_algorithm: the KNNBaseline or SVD surprise object to be used for model based approach
        Return:
            recommendations.head(number_of_recommendation): pd.Dataframe that contains the recommended movies
        """
        # Creating the surprise models for reader and dataset
        # rating_scale indicates the range of given ratings
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.ratings_df[['userId', 'movieId', 'rating']], reader)

        # Building whole trainset to train the algorithm
        train_dataset = data.build_full_trainset()
        # Building a test set from remaining part of tha dataset
        test_dataset = train_dataset.build_anti_testset()
        # Train and test the model
        recommendations = selected_algorithm.fit(train_dataset).test(test_dataset)
        # Convert the recommendations into pd.Dataframe data type
        recommendations = pd.DataFrame(recommendations, columns=['userId', 'movieId', 'trueRating', 'estimatedRating', 'USELESS COLUMN']).drop(columns='USELESS COLUMN')
        # Merge the recommendations with self.movies_df in order to get additional informations of movie title and genres
        # Sort the values in descending ortder in order to show the most similar recommendations on the top
        recommendations = pd.merge(left=recommendations[recommendations['userId'] == user_id].sort_values(by='estimatedRating', ascending=False, ignore_index=True), right=self.movies_df, on='movieId')
        return recommendations.head(number_of_recommendation)
