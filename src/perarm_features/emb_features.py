"""create embedding models for feature processing"""
import json
import os
import sys
import time
import random
import string
import argparse
import functools
import numpy as np
import pickle as pkl
from pprint import pprint
from typing import List, Union

# TF-Agent agents & networks
from tf_agents.bandits.policies import policy_utilities

from src.per_arm_rl import data_utils
from src.per_arm_rl import train_utils
from src.per_arm_rl import data_config

# logging
import logging
logging.disable(logging.WARNING)

# tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# ====================================================
# get global and per-arm features
# ====================================================

class EmbeddingModel:
    
    def __init__(
        self,
        vocab_dict: dict,
        num_oov_buckets: int,
        global_emb_size: int,
        mv_emb_size: int,
    ):
        
        self.vocab_dict = vocab_dict
        self.num_oov_buckets = num_oov_buckets
        self.global_emb_size = global_emb_size
        self.mv_emb_size = mv_emb_size
        
        # ====================================================
        # get global context (user) feature embedding models 
        # ====================================================
        
        # user_id
        user_id_input_layer = tf.keras.Input(
            name="user_id",
            shape=(1,),
            dtype=tf.string
        )
        user_id_lookup = tf.keras.layers.StringLookup(
            max_tokens=len(self.vocab_dict['user_id']) + self.num_oov_buckets,
            num_oov_indices=self.num_oov_buckets,
            mask_token=None,
            vocabulary=self.vocab_dict['user_id'],
        )(user_id_input_layer)
        user_id_embedding = tf.keras.layers.Embedding(
            # Let's use the explicit vocabulary lookup.
            input_dim=len(self.vocab_dict['user_id']) + self.num_oov_buckets,
            output_dim=self.global_emb_size
        )(user_id_lookup)
        user_id_embedding = tf.reduce_sum(user_id_embedding, axis=-2)
        
        self.user_id_model = tf.keras.Model(
            inputs=user_id_input_layer, outputs=user_id_embedding
        )
        
        # bucketized_user_age
        user_age_input_layer = tf.keras.Input(
            name="bucketized_user_age",
            shape=(1,),
            dtype=tf.float32
        )
        user_age_lookup = tf.keras.layers.IntegerLookup(
            vocabulary=self.vocab_dict['bucketized_user_age'],
            num_oov_indices=self.num_oov_buckets,
            oov_value=0,
        )(user_age_input_layer)
        user_age_embedding = tf.keras.layers.Embedding(
            # Let's use the explicit vocabulary lookup.
            input_dim=len(self.vocab_dict['bucketized_user_age']) + self.num_oov_buckets,
            output_dim=self.global_emb_size
        )(user_age_lookup)
        user_age_embedding = tf.reduce_sum(user_age_embedding, axis=-2)
        
        self.user_age_model = tf.keras.Model(
            inputs=user_age_input_layer, outputs=user_age_embedding
        )
        
        # user_occupation_text
        user_occ_input_layer = tf.keras.Input(
            name="user_occupation_text",
            shape=(1,),
            dtype=tf.string
        )
        user_occ_lookup = tf.keras.layers.StringLookup(
            max_tokens=len(self.vocab_dict['user_occupation_text']) + self.num_oov_buckets,
            num_oov_indices=self.num_oov_buckets,
            mask_token=None,
            vocabulary=self.vocab_dict['user_occupation_text'],
        )(user_occ_input_layer)
        user_occ_embedding = tf.keras.layers.Embedding(
            # Let's use the explicit vocabulary lookup.
            input_dim=len(self.vocab_dict['user_occupation_text']) + self.num_oov_buckets,
            output_dim=self.global_emb_size
        )(user_occ_lookup)
        user_occ_embedding = tf.reduce_sum(user_occ_embedding, axis=-2)
        
        self.user_occ_model = tf.keras.Model(
            inputs=user_occ_input_layer, outputs=user_occ_embedding
        )
        
        # timestamp
        user_ts_input_layer = tf.keras.Input(
            name="timestamp",
            shape=(1,),
            dtype=tf.int64
        )
        user_ts_lookup = tf.keras.layers.Discretization(
            self.vocab_dict['timestamp_buckets'].tolist()
        )(user_ts_input_layer)
        user_ts_embedding = tf.keras.layers.Embedding(
            # Let's use the explicit vocabulary lookup.
            input_dim=len(self.vocab_dict['timestamp_buckets'].tolist()) + self.num_oov_buckets,
            output_dim=self.global_emb_size
        )(user_ts_lookup)
        user_ts_embedding = tf.reduce_sum(user_ts_embedding, axis=-2)
        
        self.user_ts_model = tf.keras.Model(
            inputs=user_ts_input_layer, outputs=user_ts_embedding
        )
        
        # ====================================================
        # get perarm feature embedding models
        # ====================================================
        
        # movie_id
        mv_id_input_layer = tf.keras.Input(
            name="movie_id",
            shape=(1,),
            dtype=tf.string
        )
        mv_id_lookup = tf.keras.layers.StringLookup(
            max_tokens=len(self.vocab_dict['movie_id']) + self.num_oov_buckets,
            num_oov_indices=self.num_oov_buckets,
            mask_token=None,
            vocabulary=self.vocab_dict['movie_id'],
        )(mv_id_input_layer)
        mv_id_embedding = tf.keras.layers.Embedding(
            # Let's use the explicit vocabulary lookup.
            input_dim=len(self.vocab_dict['movie_id']) + self.num_oov_buckets,
            output_dim=self.mv_emb_size
        )(mv_id_lookup)
        mv_id_embedding = tf.reduce_sum(mv_id_embedding, axis=-2)
        
        self.mv_id_model = tf.keras.Model(
            inputs=mv_id_input_layer, outputs=mv_id_embedding
        )
        
        # movie_genres
        mv_genre_input_layer = tf.keras.Input(
            name="movie_genres",
            shape=(1,),
            dtype=tf.float32
        )
        mv_genre_lookup = tf.keras.layers.IntegerLookup(
            vocabulary=self.vocab_dict['movie_genres'],
            num_oov_indices=self.num_oov_buckets,
            oov_value=0,
        )(mv_genre_input_layer)
        mv_genre_embedding = tf.keras.layers.Embedding(
            # Let's use the explicit vocabulary lookup.
            input_dim=len(self.vocab_dict['movie_genres']) + self.num_oov_buckets,
            output_dim=self.mv_emb_size
        )(mv_genre_lookup)
        mv_genre_embedding = tf.reduce_sum(mv_genre_embedding, axis=-2)
        
        self.mv_gen_model = tf.keras.Model(
            inputs=mv_genre_input_layer, outputs=mv_genre_embedding
        )
        
        # ====================================================
        # get global context (user) sampling function
        # ====================================================
        
    # numpy.ndarray
    def _get_global_context_features(self, x) -> np.ndarray:
        """
        This function generates a single global observation vector.
        """

        _id = np.array(self.user_id_model(x['user_id']).numpy())
        _age = np.array(self.user_age_model(x['bucketized_user_age']).numpy())
        _occ = np.array(self.user_occ_model(x['user_occupation_text']).numpy())
        _ts = np.array(self.user_ts_model(x['timestamp']).numpy())

        concat = np.concatenate(
            [_id, _age, _occ, _ts], axis=-1
        ).astype(np.float32)

        return concat
        
    def _get_per_arm_features(self, x) -> np.ndarray:
        """
        This function generates a single global observation vector.
        """

        _mid = np.array(self.mv_id_model(x['movie_id']).numpy())
        _mgen = np.array(self.mv_gen_model(x['movie_genres']).numpy())

        concat = np.concatenate(
            [_mid, _mgen], axis=-1
        ).astype(np.float32)

        return concat