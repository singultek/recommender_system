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
from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from surprise import KNNBaseline, SVD
from surprise.model_selection import KFold
from tqdm import tqdm


class CollabMovies:
    def __init__(self,
                 dataset_path='./data',
                 algorithm='UserUser') -> None:
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
        self.decide_algorithm(algorithm=algorithm)
        print(self.selected_algorithm)

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

    def decide_algorithm(self, algorithm):
        self.selected_algorithm = ''
        if algorithm is not None:
            backbone = algorithm.split('-')[0]
            if backbone == 'UserUser':
                print('User-User memory based collaborative approach is selected!')
                self.selected_algorithm = backbone
            elif backbone == 'ItemItem':
                print('Item-Item memory based collaborative approach is selected!')
                self.selected_algorithm = backbone
            elif backbone == 'KNN' and len(algorithm.split('-')) == 3:
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
                    raise ValueError('The given parameters {} for KNN algorithm is not recognized. Please check the propoer format'.format(str(backbone)+'-'+str(similarity_measure)+'-'+str(user_based)))
            elif backbone == 'SVD' and len(algorithm.split('-')) == 3:
                print('SVD model based collaborative approach is selected!')
                epoch = int(algorithm.split('-')[1])
                learning_rate = float(algorithm.split('-')[2])
                self.selected_algorithm = SVD(n_epochs=epoch, lr_all=learning_rate)
            elif backbone == '':
                raise ValueError('Please give an algorithm! Algorithm cannot be left empty')
            else:
                raise ValueError('The given algorithm format {} is not recognized. Please check the propoer format'.format(str(backbone)))
        elif algorithm is None:
            raise ValueError('Please give an algorithm! Algorithm cannot be left empty')

        return self.selected_algorithm
