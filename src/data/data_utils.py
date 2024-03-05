import io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd

from typing import Dict, List, Optional, Text, Tuple

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