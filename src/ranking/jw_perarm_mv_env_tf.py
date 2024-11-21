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

# data loading
def load_movielens_data(ratings_dataset, num_users, num_items):
    # ratings = tfds.load("movielens/100k-ratings", split="train")
    ratings_matrix = np.zeros([num_users, num_items])
    
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
        ratings_matrix[
            int(row['user_id'].numpy()) - 1, 
            int(row['movie_id'].numpy()) - 1
        ] = float(row['user_rating'].numpy())
        # user_age_int.append(user_age_lookup[row['bucketized_user_age'].numpy()])
        # user_occ_int.append(user_occ_lookup[row['user_occupation_text'].numpy()])
        mov_gen_int.append(movie_gen_lookup[row['movie_genres'].numpy()])
    return tf.convert_to_tensor(ratings_matrix, dtype=tf.float32), tf.convert_to_tensor(np.array(mov_gen_int))

# Environment

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

        # self._data_matrix, self._user_age_int, self._user_occ_int, self._mov_gen_int = load_movielens_data(ratings)
        self._data_matrix, self._user_occ_int, self._mov_gen_int = load_movielens_data(ratings)
        self._num_users, self._num_movies = self._data_matrix.shape

        # Compute the SVD.
        s, u, vh = tf.linalg.svd(self._data_matrix, full_matrices=False)

        # Keep only the largest singular values.
        self._u_hat = u[:, :rank_k]
        self._s_hat = s[:rank_k]
        self._v_hat = vh[:, :rank_k]

        self._approx_ratings_matrix = tf.matmul(self._u_hat * self._s_hat,
                                                tf.transpose(self._v_hat))

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

        self._current_user_indices = tf.zeros(batch_size, dtype=np.int32)
        self._previous_user_indices = tf.zeros(batch_size, dtype=np.int32)

        self._current_movie_indices = tf.zeros([batch_size, num_actions],
                                               dtype=np.int32)
        self._previous_movie_indices = tf.zeros([batch_size, num_actions],
                                                dtype=np.int32)

        self._observation = {
            GLOBAL_KEY:
                tf.zeros([batch_size, rank_k+2], dtype=np.int32), #making space like above for dimensions
            PER_ARM_KEY:
                tf.zeros([batch_size, num_actions, rank_k+1], dtype=np.int32),
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
        
        #user section - random sample users
        sampled_user_indices_np = np.random.randint(
            self._num_users, size=self._batch_size)
        sampled_user_indices_1d = tf.convert_to_tensor(sampled_user_indices_np
                                                       , dtype=tf.int32)
        #expand dims for gather_nd - need to have indices like this [[1], [2], [5]] vs. [1, 2, 5]
        sampled_user_indices = tf.expand_dims(sampled_user_indices_1d
                                              , axis=-1)
        
        #sample feature values - gather_nd gathers the values from the randomly sampled incies
        # sampled_user_ages = tf.gather_nd(indices=sampled_user_indices
        #                                  , params=self._user_age_int)
        # sampled_user_occ = tf.gather_nd(indices=sampled_user_indices
        #                                 , params=self._user_occ_int)
        latent_user_features = tf.gather_nd(indices=sampled_user_indices
                                            , params=self._u_hat)
        
        #we concatenate these - these are our user/context features. note expand dims is needed to properly concatnate across the 1st dim
        combined_user_features = tf.concat([latent_user_features
                                                 , tf.expand_dims(sampled_user_ages, axis=-1)
                                                 , tf.expand_dims(sampled_user_occ, axis=-1)], axis=1)
    
        
        ###movie section

        sampled_movie_indices_np = np.array([
            random.sample(range(self._num_movies), self._num_actions)
            for _ in range(self._batch_size)
        ])
        sampled_movie_indices = tf.convert_to_tensor(sampled_movie_indices_np
                                                     , dtype=tf.int32)
        
        
        #expand dims for gather_nd - need to have indices like this [[1], [2], [5]] vs. [1, 2, 5]
        movie_index_vector = tf.expand_dims(tf.reshape(sampled_movie_indices, shape=[-1]), axis=-1)
        
        # movie index vector is flattened across actions now, 
        # so this will gather the genre feature values for each sampled action(movie)
        flat_genre_list = tf.gather_nd(indices=movie_index_vector, params=self._mov_gen_int) # shape of 1
        #adding actions back as a dimesions
        reshaped_genre_features = tf.reshape(flat_genre_list
                                             , shape = [self._batch_size
                                                        , self._num_actions])
        #gathering the latent movie features, again flattented at action level
        latent_movie_features = tf.gather_nd(indices=movie_index_vector
                                             , params=self._v_hat) #shape of 2
        #then we reshape the action back in
        latent_movie_features_reshaped = tf.reshape(latent_movie_features
                                                    , shape=[self._batch_size, self._num_actions, self.rank_k])
        #now that the shape is right for the latent features + the movie genre and we have dimensions = batch x action x feature dim (we concatenate at feature dim)
        current_movies = tf.concat([latent_movie_features_reshaped
                                             , tf.expand_dims(reshaped_genre_features, axis=-1)], axis=2)

        #save the indices 
        self._previous_user_indices = self._current_user_indices
        self._current_user_indices = sampled_user_indices
        self._previous_movie_indices = self._current_movie_indices
        self._current_movie_indices = sampled_movie_indices
        

        batched_observations = {
            GLOBAL_KEY:
                combined_user_features,
            PER_ARM_KEY:
                current_movies,
        }
        return batched_observations
    

    def _apply_action(self, action):
        action = tf.expand_dims(action, axis=-1)
        chosen_arm_indices = tf.gather_nd(indices=action
                                          , params=self._current_movie_indices
                                          , batch_dims = 1)
        chosen_user_moves = tf.concat([self._current_user_indices
                                       , tf.expand_dims(chosen_arm_indices, axis=-1)]
                                      , axis=1)
        return tf.gather_nd(indices=chosen_user_moves, params=self._approx_ratings_matrix)

    def _rewards_for_all_actions(self):
        broadcasted_user = tf.broadcast_to(self._previous_user_indices
                                           , [BATCH_SIZE, NUM_ACTIONS]) #broadcast the user ID across all actions
        chosen_user_movies = tf.stack([broadcasted_user      
                                       , self._previous_movie_indices]
                                      , axis=2)
        rewards_matrix = tf.gather_nd(indices=chosen_user_movies
                                      , params=self._approx_ratings_matrix)
        return rewards_matrix

    def compute_optimal_action(self):
        optimal_actions = tf.argmax(self._rewards_for_all_actions(), axis=-1)
        return tf.cast(optimal_actions, dtype=tf.int32) #needs casting

    def compute_optimal_reward(self):
        return np.max(self._rewards_for_all_actions(), axis=-1)