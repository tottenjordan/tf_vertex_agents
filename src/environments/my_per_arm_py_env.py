# Copyright 2021 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
"""Class implementation of the per-arm MovieLens Bandit environment."""
from __future__ import absolute_import
import os
# import gin
import random
import numpy as np
from typing import Optional, Text
import logging

# tensorflow
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

# tf-agents
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.bandits.environments import bandit_py_environment
from tf_agents.bandits.environments import dataset_utilities
from tf_agents.bandits.specs import utils as bandit_spec_utils

# this repo
from src.data import data_utils as data_utils

GLOBAL_KEY = bandit_spec_utils.GLOBAL_FEATURE_KEY
PER_ARM_KEY = bandit_spec_utils.PER_ARM_FEATURE_KEY

# @gin.configurable
class MyMovieLensPerArmPyEnvironment(bandit_py_environment.BanditPyEnvironment):
    """Implements the per-arm version of the MovieLens Bandit environment.

    This environment implements the MovieLens 100K dataset, available at:
    https://www.kaggle.com/prajitdatta/movielens-100k-dataset

    This dataset contains 100K ratings from 943 users on 1682 items.
    This environment computes a low-rank matrix factorization (using SVD) of the
    data matrix `A`, such that: `A ~= U * Sigma * V^T`.

    The environment uses the rows of `U` as global (or user) features, and the
    rows of `V` as per-arm (or movie) features.

    The reward of recommending movie `v` to user `u` is `u * Sigma * v^T`.
    """

    def __init__(
        self
        , project_number
        , bucket_name: str
        , data_gcs_prefix: str
        , user_age_lookup_dict: dict
        , user_occ_lookup_dict: dict
        # , movie_gen_lookup_dict: dict
        , num_users: int = 943
        , num_movies: int = 1682
        , rank_k: int = 2
        , batch_size: int = 10
        , num_actions: int = 100
        , name: Optional[Text] = 'movielens_per_arm'
    ):
        """Initializes the Per-arm MovieLens Bandit environment.

        Args:
          data_dir: (string) Directory where the data lies (in text form).
          rank_k : (int) Which rank to use in the matrix factorization. This will
            also be the feature dimension of both the user and the movie features.
          batch_size: (int) Number of observations generated per call.
          num_actions: (int) How many movies to choose from per round.
          name: (string) The name of this environment instance.
        """
        self.project_number = project_number
        self.bucket_name = bucket_name
        self.data_gcs_prefix = data_gcs_prefix
        self._batch_size = batch_size
        self._num_actions = num_actions
        self._rank_k = rank_k
        self.num_users = num_users
        self.num_movies = num_movies
        self.user_age_lookup_dict = user_age_lookup_dict
        self.user_occ_lookup_dict = user_occ_lookup_dict
        # self.movie_gen_lookup_dict = movie_gen_lookup_dict
        
        # =============================================
        # set GCP clients
        # =============================================
        from google.cloud import aiplatform as vertex_ai
        from google.cloud import storage
        
        # project_number = os.environ["CLOUD_ML_PROJECT_ID"]
        storage_client = storage.Client(project=self.project_number)
        vertex_ai.init(
            project=self.project_number,
            location='us-central1',
            # experiment=args.experiment_name
        )
        
        # =============================================
        # get TF dataset
        # =============================================
        logging.info("Creating TRAIN dataset...")
        logging.info(f'Path to TRAIN files: gs://{self.bucket_name}/{self.data_gcs_prefix}')
        
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        
        train_files = []
        for blob in storage_client.list_blobs(
            f"{self.bucket_name}", 
            prefix=f'{self.data_gcs_prefix}/', 
            # delimiter="/"
        ):
            if '.tfrecord' in blob.name:
                train_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))
                
        logging.info(f'train_files: {train_files}')
        
        train_dataset = tf.data.TFRecordDataset(train_files)
        train_dataset = train_dataset.map(data_utils._parse_function)
        
        self.dataset = train_dataset
        # =============================================
        # load data & compute ratings matrix
        # =============================================
        self._data_matrix, self._user_age_int, self._user_occ_int = data_utils.load_movielens_ratings(
            ratings_dataset = train_dataset
            , num_users = self.num_users
            , num_movies = self.num_movies
            , user_age_lookup_dict = self.user_age_lookup_dict
            , user_occ_lookup_dict = self.user_occ_lookup_dict
            # , movie_gen_lookup_dict = self.movie_gen_lookup_dict
        )
        self._num_users, self._num_movies = self._data_matrix.shape
        
        # Compute the SVD.
        # TODO - compute SVD in tf - see: https://www.tensorflow.org/api_docs/python/tf/linalg/svd
        u, s, vh = np.linalg.svd(self._data_matrix, full_matrices=False)
        
        # Keep only the largest singular values.
        self._u_hat = u[:, :self._rank_k].astype(np.float32)
        self._s_hat = s[:self._rank_k].astype(np.float32)
        self._v_hat = np.transpose(vh[:self._rank_k]).astype(np.float32)

        self._approx_ratings_matrix = np.matmul(
            self._u_hat * self._s_hat
            , np.transpose(self._v_hat)
        )
        
        # =============================================
        # action spec
        # =============================================
        self._action_spec = array_spec.BoundedArraySpec(
            shape=()
            , dtype=np.int32
            , minimum=0
            , maximum=self._num_actions - 1
            , name='action'
        )
        
        # =============================================
        # observation spec
        # =============================================
        observation_spec = {
            GLOBAL_KEY:
                array_spec.ArraySpec(shape=[self._rank_k + 2], dtype=np.float32),     # creating +space for user age and occupation
            PER_ARM_KEY:
                array_spec.ArraySpec(
                    shape=[self._num_actions, self._rank_k + 1], dtype=np.float32),   # creating +1 space for movie genre
        }
        self._time_step_spec = ts.time_step_spec(observation_spec)

        self._current_user_indices = np.zeros(self._batch_size, dtype=np.int32)
        self._previous_user_indices = np.zeros(self._batch_size, dtype=np.int32)

        self._current_movie_indices = np.zeros(
            [self._batch_size, self._num_actions], dtype=np.int32
        )
        self._previous_movie_indices = np.zeros(
            [self._batch_size, self._num_actions], dtype=np.int32
        )
        
        self._observation = {
            GLOBAL_KEY:
                np.zeros([self._batch_size, self._rank_k + 2], dtype=np.int32), #making space like above for dimensions
            PER_ARM_KEY:
                np.zeros([self._batch_size, self._num_actions, self._rank_k + 1], dtype=np.int32),
        }
        
        # =============================================
        # super init
        # =============================================
        super(MyMovieLensPerArmPyEnvironment, self).__init__(
            observation_spec, self._action_spec, name=name
        )
        
    # =============================================
    # batched observations
    # =============================================
    @property
    def batch_size(self):
        return self._batch_size

    @property
    def batched(self):
        return True

    def _observe(self):
        sampled_user_indices = np.random.randint(
            self._num_users, size=self._batch_size
        )
        self._previous_user_indices = self._current_user_indices
        self._current_user_indices = sampled_user_indices

        sampled_movie_indices = np.array(
            [
                random.sample(range(self._num_movies), self._num_actions)
                for _ in range(self._batch_size)
            ]
        )
        sampled_user_ages = self._user_age_int[sampled_user_indices]
        sampled_user_occ = self._user_occ_int[sampled_user_indices]
        combined_user_features = np.concatenate(
            (
                self._u_hat[sampled_user_indices]
                , sampled_user_ages.reshape(-1,1)
                , sampled_user_occ.reshape(-1,1)
            )
            , axis=1
        )
        # current_users = combined_user_features.reshape([self.batch_size, self._rank_k+2])

        movie_index_vector = sampled_movie_indices.reshape(-1)
        # print(movie_index_vector.shape)

        # flat_genre_list = self._mov_gen_int[movie_index_vector] # shape of 1
        flat_genre_list = self._user_occ_int[movie_index_vector] # shape of 1
        flat_movie_list = self._v_hat[movie_index_vector]       # shape of 2

        combined_movie_features = np.concatenate(
            (
                flat_movie_list
                , flat_genre_list.reshape(-1,1)
            )
            , axis=1
        )

        current_movies = combined_movie_features.reshape(
            [self._batch_size, self._num_actions, self._rank_k + 1]
        )

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
        chosen_arm_indices = self._current_movie_indices[
            range(self._batch_size), action
        ]
        return self._approx_ratings_matrix[
            self._current_user_indices, chosen_arm_indices
        ]

    def _rewards_for_all_actions(self):
        rewards_matrix = self._approx_ratings_matrix[
            np.expand_dims(self._previous_user_indices, axis=-1)
            , self._previous_movie_indices
        ]
        return rewards_matrix

    def compute_optimal_action(self):
        return np.argmax(self._rewards_for_all_actions(), axis=-1)

    def compute_optimal_reward(self):
        return np.max(self._rewards_for_all_actions(), axis=-1)