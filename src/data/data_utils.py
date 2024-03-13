import io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd

from typing import Dict, List, Optional, Text, Tuple
from google.cloud import aiplatform, bigquery, storage

# tensorflow
import tensorflow as tf

# logging
import logging
logging.disable(logging.WARNING)

import warnings
warnings.filterwarnings('ignore')

# this repo
from . import data_config as data_config

# features
USER_FEATURE_NAMES = [
    'user_id',
    'user_age',
    'user_occupation_text',
    'target_rating_timestamp',
    'user_zip_code',
    'user_gender',
]
MOVIE_FEATURE_NAMES = [
    'target_movie_id',
    'target_movie_title',
    'target_movie_year',
    'target_movie_genres',
    # 'target_movie_tags',
]
# TARGET_FEATURE_NAME = "user_rating"
TARGET_FEATURE_NAME = "target_movie_rating"

# ==========================================
# gcp helpers
# ==========================================
def download_blob(
    project_id, 
    bucket_name, 
    source_blob_name, 
    destination_file_name
):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )

# ==========================================
# parsing functions
# ==========================================
    
feature_description = {
    # # context sequence item features
    # 'context_movie_id': tf.io.FixedLenFeature(shape=(MAX_CONTEXT_LENGTH), dtype=tf.string),
    # 'context_movie_rating': tf.io.FixedLenFeature(shape=(MAX_CONTEXT_LENGTH), dtype=tf.float32),
    # 'context_rating_timestamp': tf.io.FixedLenFeature(shape=(MAX_CONTEXT_LENGTH), dtype=tf.int64),
    # 'context_movie_genre': tf.io.FixedLenFeature(shape=(MAX_GENRE_LENGTH), dtype=tf.string),
    # 'context_movie_year': tf.io.FixedLenFeature(shape=(MAX_CONTEXT_LENGTH), dtype=tf.int64),
    # 'context_movie_title': tf.io.FixedLenFeature(shape=(MAX_CONTEXT_LENGTH), dtype=tf.string),

    # target/label item features
    'target_movie_id': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
    'target_movie_rating': tf.io.FixedLenFeature(shape=(), dtype=tf.float32),
    'target_rating_timestamp': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
    'target_movie_genres': tf.io.FixedLenFeature(shape=(data_config.MAX_GENRE_LENGTH), dtype=tf.string),
    'target_movie_year': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
    'target_movie_title': tf.io.FixedLenFeature(shape=(), dtype=tf.string),

    # user - global context features
    'user_id': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
    'user_gender': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
    'user_age': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
    'user_occupation_text': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
    'user_zip_code': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
}

def _parse_function(example_proto):
    return tf.io.parse_single_example(
        example_proto, feature_description
    )

def full_parse(data):
    # used for interleave - takes tensors and returns a tf.dataset
    data = tf.data.TFRecordDataset(data)
    return data

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
    # , movie_gen_lookup_dict: dict
):
    """
    > loads (wide) movielens ratings data 
    > returns ratings matrix
    """
    ratings_matrix = np.zeros([num_users, num_movies])
    
    local_data = ratings_dataset.map(
        lambda x: {
            'user_id': x['user_id']
            ,'target_movie_id':  x['target_movie_id']
            ,'target_movie_rating':  x['target_movie_rating']
            ,'user_age': x['user_age']
            ,'user_occupation_text': x['user_occupation_text']
            # ,'movie_genres': x['movie_genres'][0]
        }
    )
    user_age_int = []
    user_occ_int = []
    mov_gen_int = []
    
    for row in local_data:
        ratings_matrix[
            int(row['user_id'].numpy()) - 1
            , int(row['target_movie_id'].numpy()) - 1
        ] = float(row['target_movie_rating'].numpy())
        
        user_age_int.append(
            # float(user_age_lookup_dict[row['user_age'].numpy()]) + .0001
            float(row['user_age'].numpy()) + .0001
        )
        user_occ_int.append(
            float(user_occ_lookup_dict[row['user_occupation_text'].numpy()]) + .0001
        )
        # mov_gen_int.append(
        #     float(movie_gen_lookup_dict[row['movie_genres'].numpy()]) + .0001
        # ) 
    return ratings_matrix, np.array(user_age_int), np.array(user_occ_int)