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

from google.cloud import aiplatform as vertex_ai
from google.cloud import storage

# ============================================
# features
# ============================================

# MAX_LIST_LENGTH = 3
MAX_LIST_LENGTH = 5

def get_all_features():
    
    feats = {
        # user - global context features
        'user_id': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        'user_rating': tf.io.FixedLenFeature(shape=(), dtype=tf.float32),
        'bucketized_user_age': tf.io.FixedLenFeature(shape=(), dtype=tf.float32),
        'user_occupation_text': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        # 'user_occupation_label': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
        # 'user_zip_code': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        # 'user_gender': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        'timestamp': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),

        # movie - per arm features
        'movie_id': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        # 'movie_title': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        'movie_genres': tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64),
    }
    
    return feats

def get_all_lw_features(MAX_LIST_LENGTH):
    '''
    listwise features
    '''
    feats = {
        'user_id': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        "movie_id": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_LIST_LENGTH,)), 
        "movie_genres": tf.io.FixedLenFeature(dtype=tf.int64, shape=(MAX_LIST_LENGTH,)), 
        "user_rating": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_LIST_LENGTH,))
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
    
    example = tf.io.parse_example(
        example,
        feats
        # features=feats
    )
    return example

def parse_lw_tfrecord(example):
    """
    Reads a serialized example from GCS and converts to tfrecord
    """
    feats = get_all_lw_features(MAX_LIST_LENGTH)
    
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

# ================================================
# converting features to `tf.train.Example` proto
# ================================================

def _bytes_feature(value):
    """
    Get byte features
    """
    # value = tf.io.serialize_tensor(value)
    # value = value.numpy()
    if type(value) == list:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[i.numpy() for i in [value]]))
    
def _bytes_array_feature(value):
    """
    Get byte features
    TODO - consolidate with above
    """
    # value = tf.io.serialize_tensor(value)
    # value = value.numpy()
    if type(value) == np.ndarray:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[i.numpy() for i in [value]]))
    
# def _bytes_feature_v2(value):
#     """Returns a bytes_list from a string / byte."""
#     # if isinstance(value, type(tf.constant(0))):
#     value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
#     # return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[i for i in value]))

def _bytes_feature_v2(value):
    """
    Get byte features
    """
    # value = tf.io.serialize_tensor(value)
    # value = value.numpy()
    if type(value) == list:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[i.numpy() for i in value]))

def _int64_feature(value):
    """
    Get int64 feature
    """
    if type(value) == list:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v) for v in value]))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
def _int64_list_feature(value):
    """
    Get int64 list feature
    """
    # value = value.numpy().tolist() #[0]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v) for v in value]))

def _string_array(value, shape=1):
    """
    Returns a bytes_list from a string / byte.
    """
    value = value.numpy() #[0] # .tolist()[0]
    if type(value) == list:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(v).encode('utf-8') for v in value]))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value).encode('utf-8')]))

def _float_feature(value, shape=1):
    """
    Returns a float_list from a float / double.
    """
    if type(value) == np.ndarray: #list:
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    else:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    
def _float_list_feature(value, shape=1):
    """
    Returns a float_list from a float / double.
    TODO - consolidate with above
    """
    # value = value.numpy()

    if type(value) == np.ndarray: # list
        return tf.train.Feature(float_list=tf.train.FloatList(value=[int(v) for v in [value]]))
    else:
        vector = np.vectorize(float)
        return tf.train.Feature(float_list=tf.train.FloatList(value=vector(value)))
        # return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    

def build_example(data) -> tf.train.Example:
    """
    Returns: A `tf.train.Example` object holding the same data as `data_row`.
    """
    feature = {
        # user - global context features 
        "user_id": _bytes_feature(data['user_id']),
        "user_rating": _float_feature(data['user_rating']),
        "bucketized_user_age": _float_feature(data['bucketized_user_age']),
        "user_occupation_text": _bytes_feature(data['user_occupation_text']),
        # "user_occupation_label": _int64_feature(data['user_occupation_label']),
        # "user_zip_code": _bytes_feature(data['user_zip_code']),
        # "user_gender": _string_array(data['user_gender']),
        "timestamp": _int64_feature(data['timestamp']),
        
        # movie - per arm features
        "movie_id": _bytes_feature(data['movie_id']),
        # "movie_title": _bytes_feature(data['movie_title']),
        "movie_genres": _int64_list_feature(data['movie_genres']),
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )
    return example_proto

def build_list_wise_example(data) -> tf.train.Example:
    """
    Returns: A `tf.train.Example` object holding the same data as `data_row`.
    """
    feature = {
        # user - global context features 
        "user_id": _bytes_feature(data['user_id']),
        "user_rating": _float_list_feature(data['user_rating']),
        
        # movie - per arm features
        "movie_id": _bytes_feature_v2(data['movie_id']),
        "movie_genres": _int64_list_feature(data['movie_genres']),
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )
    return example_proto # example_proto.SerializeToString()


# ============================================
# TF lookup dictionary
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
# TF-Record Writer
# ============================================
def write_tfrecords(tfrecord_file, dataset, list_wise=False):
    if list_wise:
        with tf.io.TFRecordWriter(tfrecord_file) as writer:
            for data_row in dataset:
                example = build_list_wise_example(data_row)
                writer.write(example.SerializeToString())
    else:
        with tf.io.TFRecordWriter(tfrecord_file) as writer:
            for data_row in dataset:
                example = build_example(data_row)
                writer.write(example.SerializeToString())

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

# ====================================================
# upload object to GCS
# ====================================================

# upload files to Google Cloud Storage
def upload_blob(
    project_id, 
    bucket_name, 
    source_file_name, 
    destination_blob_name
):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name" (no 'gs://')
    # source_file_name = "local/path/to/file" (file to upload)
    # destination_blob_name = "folder/paths-to/storage-object-name"
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to gs://{bucket_name}/{destination_blob_name}."
    )
    
def download_blob(project_id, bucket_name, source_blob_name, destination_file_name):
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