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


def initialize_collab(ratings):
    total_number_of_ratings = ratings.shape[0]
    total_number_of_movies = ratings[['movieId']].drop_duplicates(['movieId']).shape[0]
    total_number_of_users = ratings[['userId']].drop_duplicates(['userId']).shape[0]

    print('\nColumns/Features of Dataset: ', list(ratings.columns))
    print('Total Number of Ratings : ', total_number_of_ratings)
    print('Total Number of Movies: ', total_number_of_movies)
    print('Total Number of Users: ', total_number_of_users)
    return
