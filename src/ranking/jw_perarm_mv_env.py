"""Class implementation of the per-arm MovieLens Bandit environment."""
from __future__ import absolute_import

import random
from typing import Optional, Text
import gin
import numpy as np

from tf_agents.bandits.environments import bandit_py_environment
from tf_agents.bandits.environments import dataset_utilities
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


GLOBAL_KEY = bandit_spec_utils.GLOBAL_FEATURE_KEY
PER_ARM_KEY = bandit_spec_utils.PER_ARM_FEATURE_KEY

def load_movielens_data(ratings_dataset):
    # ratings = tfds.load("movielens/100k-ratings", split="train")
    ratings_matrix = np.zeros([MOVIELENS_NUM_USERS, MOVIELENS_NUM_MOVIES])
    
    local_data = ratings_dataset.map(
        lambda x: {
            'user_id': x['user_id'],
            'movie_id':  x['movie_id'],
            'user_rating':  x['user_rating'],
            # 'bucketized_user_age': x['bucketized_user_age'],
            # 'user_occupation_text': x['user_occupation_text'],
            'movie_genres': x['movie_genres'][0],
        }
    )
    
    # user_age_int = []
    # user_occ_int = []
    mov_gen_int = []
    
    for row in local_data:
        ratings_matrix[int(row['user_id'].numpy()) - 1, int(row['movie_id'].numpy()) - 1] = float(row['user_rating'].numpy())
        # user_age_int.append(float(user_age_lookup[row['bucketized_user_age'].numpy()])+.0001)
        # user_occ_int.append(float(user_occ_lookup[row['user_occupation_text'].numpy()])+.0001)
        mov_gen_int.append(float(movie_gen_lookup[row['movie_genres'].numpy()])+.0001) 
    return ratings_matrix, np.array(user_age_int), np.array(user_occ_int), np.array(mov_gen_int)


# @gin.configurable
class MovieLensPerArmPyEnvironment(bandit_py_environment.BanditPyEnvironment):
    """Implements the per-arm version of the MovieLens Bandit environment.

    This environment implements the MovieLens 100K dataset, available at:
    https://www.kaggle.com/prajitdatta/movielens-100k-dataset

    This dataset contains 100K ratings from 943 users on 1682 items.
    This csv list of:
    user id | item id | rating | timestamp.
    This environment computes a low-rank matrix factorization (using SVD) of the
    data matrix `A`, such that: `A ~= U * Sigma * V^T`.

    The environment uses the rows of `U` as global (or user) features, and the
    rows of `V` as per-arm (or movie) features.

    The reward of recommending movie `v` to user `u` is `u * Sigma * v^T`.
    """

    def __init__(self,
               dataset: str, # = ratings,
               rank_k: int = 2,
               batch_size: int = 10,
               num_actions: int = 100,
               name: Optional[Text] = 'movielens_per_arm'):
        """Initializes the Per-arm MovieLens Bandit environment.

        Args:
          data_dir: (string) Directory where the data lies (in text form).
          rank_k : (int) Which rank to use in the matrix factorization. This will
            also be the feature dimension of both the user and the movie features.
          batch_size: (int) Number of observations generated per call.
          num_actions: (int) How many movies to choose from per round.
          csv_delimiter: (string) The delimiter to use in loading the data csv file.
          name: (string) The name of this environment instance.
        """
        self._batch_size = batch_size
        self._num_actions = num_actions
        self.rank_k = rank_k

        # Compute the matrix factorization.
        # self._data_matrix = dataset_utilities.load_movielens_data(
        #     data_dir, delimiter=csv_delimiter)

        self._data_matrix, self._user_age_int, self._user_occ_int, self._mov_gen_int = load_movielens_data(ratings)
        self._num_users, self._num_movies = self._data_matrix.shape

        # Compute the SVD.
        u, s, vh = np.linalg.svd(self._data_matrix, full_matrices=False)

        # Keep only the largest singular values.
        self._u_hat = u[:, :rank_k].astype(np.float32)
        self._s_hat = s[:rank_k].astype(np.float32)
        self._v_hat = np.transpose(vh[:rank_k]).astype(np.float32)

        self._approx_ratings_matrix = np.matmul(self._u_hat * self._s_hat,
                                                np.transpose(self._v_hat))

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,
            maximum=num_actions - 1,
            name='action')
        observation_spec = {
            GLOBAL_KEY:
                array_spec.ArraySpec(shape=[rank_k+2], dtype=np.float32), #creating +space for user age and occupation
            PER_ARM_KEY:
                array_spec.ArraySpec(
                    shape=[num_actions, rank_k+1], dtype=np.float32), #creating +1 space for movie genre
        }
        self._time_step_spec = ts.time_step_spec(observation_spec)

        self._current_user_indices = np.zeros(batch_size, dtype=np.int32)
        self._previous_user_indices = np.zeros(batch_size, dtype=np.int32)

        self._current_movie_indices = np.zeros([batch_size, num_actions],
                                               dtype=np.int32)
        self._previous_movie_indices = np.zeros([batch_size, num_actions],
                                                dtype=np.int32)

        self._observation = {
            GLOBAL_KEY:
                np.zeros([batch_size, rank_k+2], dtype=np.int32), #making space like above for dimensions
            PER_ARM_KEY:
                np.zeros([batch_size, num_actions, rank_k+1], dtype=np.int32),
        }

        super(MovieLensPerArmPyEnvironment, self).__init__(
            observation_spec, self._action_spec, name=name)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def batched(self):
        return True

    def _observe(self):
        sampled_user_indices = np.random.randint(
            self._num_users, size=self._batch_size)
        self._previous_user_indices = self._current_user_indices
        self._current_user_indices = sampled_user_indices

        sampled_movie_indices = np.array([
            random.sample(range(self._num_movies), self._num_actions)
            for _ in range(self._batch_size)
        ])
        sampled_user_ages = self._user_age_int[sampled_user_indices]
        sampled_user_occ = self._user_occ_int[sampled_user_indices]
        combined_user_features = np.concatenate((self._u_hat[sampled_user_indices]
                                                 , sampled_user_ages.reshape(-1,1)
                                                 , sampled_user_occ.reshape(-1,1)), axis=1)
        # current_users = combined_user_features.reshape([self._batch_size, self.rank_k+2])
        
        movie_index_vector = sampled_movie_indices.reshape(-1)
        print(movie_index_vector.shape)
        flat_genre_list = self._mov_gen_int[movie_index_vector] #shape of 1
        flat_movie_list = self._v_hat[movie_index_vector] #shape of 2
        combined_movie_features = np.concatenate((flat_movie_list,flat_genre_list.reshape(-1,1)), axis=1)
        current_movies = combined_movie_features.reshape(
            [self._batch_size, self._num_actions, self.rank_k+1])

        self._previous_movie_indices = self._current_movie_indices
        self._current_movie_indices = sampled_movie_indices

        batched_observations = {
            GLOBAL_KEY:
                tf.convert_to_tensor(combined_user_features, dtype=tf.float32),
            PER_ARM_KEY:
                tf.convert_to_tensor(current_movies, dtype=tf.float32),
        }
        return batched_observations

    def _apply_action(self, action):
        chosen_arm_indices = self._current_movie_indices[range(self._batch_size),
                                                         action]
        return self._approx_ratings_matrix[self._current_user_indices,
                                           chosen_arm_indices]

    def _rewards_for_all_actions(self):
        rewards_matrix = self._approx_ratings_matrix[
            np.expand_dims(self._previous_user_indices, axis=-1),
            self._previous_movie_indices]
        return rewards_matrix

    def compute_optimal_action(self):
        return np.argmax(self._rewards_for_all_actions(), axis=-1)

    def compute_optimal_reward(self):
        return np.max(self._rewards_for_all_actions(), axis=-1)