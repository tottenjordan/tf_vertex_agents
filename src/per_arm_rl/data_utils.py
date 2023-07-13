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

import os
import numpy as np
from typing import Dict

import tensorflow as tf


EMBEDDING_SIZE = 128

# ============================================
# features
# ============================================
# DEFAULT_FEATURE_MAP = {
#     # user - global context features
#     'user_id': tf.io.FixedLenSequenceFeature([], tf.string),
#     'user_rating': tf.io.FixedLenSequenceFeature([], tf.float32),
#     'bucketized_user_age': tf.io.FixedLenSequenceFeature([], tf.float32),
#     'user_occupation_text': tf.io.FixedLenSequenceFeature([], tf.string),
#     # 'user_occupation_label': tf.io.FixedLenSequenceFeature([], tf.int64),
#     'timestamp': tf.io.FixedLenSequenceFeature([], tf.int64),
#     # 'user_zip_code': tf.io.FixedLenSequenceFeature([], tf.string),
#     # 'user_gender': tf.io.FixedLenSequenceFeature([], tf.bool),
    
#     # movie - per arm features
#     'movie_id': tf.io.FixedLenSequenceFeature([], tf.string),
#     'movie_title': tf.io.FixedLenSequenceFeature([], tf.string),
#     'movie_genres': tf.io.FixedLenSequenceFeature([], tf.int64),
# }

def get_all_features():
    
    feats = {
        # user - global context features
        'user_id': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        'user_rating': tf.io.FixedLenFeature(shape=(), dtype=tf.float32),
        'bucketized_user_age': tf.io.FixedLenFeature(shape=(), dtype=tf.float32),
        'user_occupation_text': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        # 'user_occupation_label': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
        'timestamp': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
        # 'user_zip_code': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        # 'user_gender': tf.io.FixedLenFeature(shape=(), dtype=tf.bool),

        # movie - per arm features
        'movie_id': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        # 'movie_title': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        'movie_genres': tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64),
    }
    
    return feats 

# ============================================
# tf data parsing functions
# ============================================
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

def parse_tfrecord(example):
    """
    Reads a serialized example from GCS and converts to tfrecord
    """
    feats = get_all_features()
    
    # example = tf.io.parse_single_example(
    example = tf.io.parse_example(
        example,
        feats
        # features=feats
    )
    return example

# data loading and parsing
def full_parse(data):
    # used for interleave - takes tensors and returns a tf.dataset
    data = tf.data.TFRecordDataset(data)
    return data

# ============================================
# Helper function for TF lookup dictionary
# ============================================

def get_dictionary_lookup_by_tf_data_key(key, dataset) -> Dict:
    tensor = dataset.map(lambda x: x[key])
    unique_elems = set()
    for x in tensor:
        val = x.numpy()
        if type(val) is np.ndarray: # if multi dimesnional only grab first one
            val = val[0]
        unique_elems.add(val)
    
    #return a dictionary of keys by integer values for the feature space
    return {val: i for i, val in enumerate(unique_elems)}

# ============================================
# load movielens
# ============================================
def load_movielens_ratings(
    ratings_dataset
    , num_users: int
    , num_movies: int
    , user_age_lookup_dict: dict
    , user_occ_lookup_dict: dict
    , movie_gen_lookup_dict: dict
):
    """
    > loads (wide) movielens ratings data 
    > returns ratings matrix
    """
    # ratings = tfds.load("movielens/100k-ratings", split="train")
    ratings_matrix = np.zeros([num_users, num_movies])
    
    local_data = ratings_dataset.map(
        lambda x: {
            'user_id': x['user_id']
            ,'movie_id':  x['movie_id']
            ,'user_rating':  x['user_rating']
            ,'bucketized_user_age': x['bucketized_user_age']
            ,'user_occupation_text': x['user_occupation_text']
            ,'movie_genres': x['movie_genres'][0]
        }
    )
    user_age_int = []
    user_occ_int = []
    mov_gen_int = []
    
    for row in local_data:
        ratings_matrix[
            int(row['user_id'].numpy()) - 1
            , int(row['movie_id'].numpy()) - 1
        ] = float(row['user_rating'].numpy())
        
        user_age_int.append(
            float(user_age_lookup_dict[row['bucketized_user_age'].numpy()]) + .0001
        )
        user_occ_int.append(
            float(user_occ_lookup_dict[row['user_occupation_text'].numpy()]) + .0001
        )
        mov_gen_int.append(
            float(movie_gen_lookup_dict[row['movie_genres'].numpy()]) + .0001
        ) 
    return ratings_matrix, np.array(user_age_int), np.array(user_occ_int), np.array(mov_gen_int)
