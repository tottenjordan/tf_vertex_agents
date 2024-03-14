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
        max_genre_length: int,
    ):
        
        self.vocab_dict = vocab_dict
        self.num_oov_buckets = num_oov_buckets
        self.global_emb_size = global_emb_size
        self.mv_emb_size = mv_emb_size
        self.max_genre_length = max_genre_length
        
        # ====================================================
        # get global context (user) feature embedding models 
        # ====================================================
        
        # ====================================================
        # user_id
        # ====================================================
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
        
        # ====================================================
        # user_age | bucketized_user_age
        # ====================================================
        user_age_input_layer = tf.keras.Input(
            name="user_age", # bucketized_user_age
            shape=(1,),
            dtype=tf.int64
        )
        user_age_lookup = tf.keras.layers.IntegerLookup(
            vocabulary=self.vocab_dict['user_age_vocab'], # bucketized_user_age
            num_oov_indices=self.num_oov_buckets,
            oov_value=0,
        )(user_age_input_layer)
        user_age_embedding = tf.keras.layers.Embedding(
            # Let's use the explicit vocabulary lookup.
            input_dim=len(self.vocab_dict['user_age_vocab']) + self.num_oov_buckets,
            output_dim=self.global_emb_size
        )(user_age_lookup)
        user_age_embedding = tf.reduce_sum(user_age_embedding, axis=-2)
        
        self.user_age_model = tf.keras.Model(
            inputs=user_age_input_layer, outputs=user_age_embedding
        )
        
        # ====================================================
        # user_occupation_text
        # ====================================================
        user_occ_input_layer = tf.keras.Input(
            name="user_occupation_text",
            shape=(1,),
            dtype=tf.string
        )
        user_occ_lookup = tf.keras.layers.StringLookup(
            max_tokens=len(self.vocab_dict['user_occ_vocab']) + self.num_oov_buckets,
            num_oov_indices=self.num_oov_buckets,
            mask_token=None,
            vocabulary=self.vocab_dict['user_occ_vocab'],
        )(user_occ_input_layer)
        user_occ_embedding = tf.keras.layers.Embedding(
            # Let's use the explicit vocabulary lookup.
            input_dim=len(self.vocab_dict['user_occ_vocab']) + self.num_oov_buckets,
            output_dim=self.global_emb_size
        )(user_occ_lookup)
        user_occ_embedding = tf.reduce_sum(user_occ_embedding, axis=-2)
        
        self.user_occ_model = tf.keras.Model(
            inputs=user_occ_input_layer, outputs=user_occ_embedding
        )
        
        # ====================================================
        # timestamp
        # ====================================================
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
        # user_zip_code
        # ====================================================
        user_zip_input_layer = tf.keras.Input(
            name="user_zip_code",
            shape=(1,),
            dtype=tf.string
        )
        user_zip_lookup = tf.keras.layers.StringLookup(
            max_tokens=len(self.vocab_dict['user_zip_vocab']) + self.num_oov_buckets,
            num_oov_indices=self.num_oov_buckets,
            mask_token=None,
            vocabulary=self.vocab_dict['user_zip_vocab'],
        )(user_zip_input_layer)
        user_zip_embedding = tf.keras.layers.Embedding(
            # Let's use the explicit vocabulary lookup.
            input_dim=len(self.vocab_dict['user_zip_vocab']) + self.num_oov_buckets,
            output_dim=self.global_emb_size
        )(user_zip_lookup)
        user_zip_embedding = tf.reduce_sum(user_zip_embedding, axis=-2)
        
        self.user_zip_model = tf.keras.Model(
            inputs=user_zip_input_layer, outputs=user_zip_embedding
        )
        
        # ====================================================
        # user_gender
        # ====================================================
        user_gender_input_layer = tf.keras.Input(
            name="user_gender",
            shape=(1,),
            dtype=tf.string
        )
        user_gender_lookup = tf.keras.layers.StringLookup(
            max_tokens=len(self.vocab_dict['user_gender_vocab']) + self.num_oov_buckets,
            num_oov_indices=self.num_oov_buckets,
            mask_token=None,
            vocabulary=self.vocab_dict['user_gender_vocab'],
        )(user_gender_input_layer)
        user_gender_embedding = tf.keras.layers.Embedding(
            # Let's use the explicit vocabulary lookup.
            input_dim=len(self.vocab_dict['user_gender_vocab']) + self.num_oov_buckets,
            output_dim=self.global_emb_size
        )(user_gender_lookup)
        user_gender_embedding = tf.reduce_sum(user_gender_embedding, axis=-2)
        
        self.user_gender_model = tf.keras.Model(
            inputs=user_gender_input_layer, outputs=user_gender_embedding
        )
        
        # ====================================================
        #
        # get perarm feature embedding models
        #
        # ====================================================
        
        # ====================================================
        # movie_id
        # ====================================================
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
        
        # ====================================================
        # movie_title
        # ====================================================
        mv_title_input_layer = tf.keras.Input(
            name="movie_title",
            shape=(1,),
            dtype=tf.string
        )
        mv_title_text = tf.keras.layers.TextVectorization(
            # max_tokens=max_tokens, 
            ngrams=2, 
            vocabulary=vocab_dict['movie_title'],
        )(mv_title_input_layer)
        mv_title_embedding = tf.keras.layers.Embedding(
            # Let's use the explicit vocabulary lookup.
            input_dim=len(self.vocab_dict['movie_title']) + self.num_oov_buckets,
            output_dim=self.mv_emb_size
        )(mv_title_text)
        mv_title_embedding = tf.reduce_sum(mv_title_embedding, axis=-2)
        
        self.mv_title_model = tf.keras.Model(
            inputs=mv_title_input_layer, outputs=mv_title_embedding
        )
        
        # ====================================================
        # movie year
        # ====================================================
        mv_year_input_layer = tf.keras.Input(
            name="movie_year",
            shape=(1,),
            dtype=tf.int64
        )
        mv_year_lookup = tf.keras.layers.IntegerLookup(
            vocabulary=self.vocab_dict['movie_year'],
            num_oov_indices=self.num_oov_buckets,
            oov_value=0,
        )(mv_year_input_layer)
        mv_year_embedding = tf.keras.layers.Embedding(
            # Let's use the explicit vocabulary lookup.
            input_dim=len(self.vocab_dict['movie_year']) + self.num_oov_buckets,
            output_dim=self.mv_emb_size
        )(mv_year_lookup)
        mv_year_embedding = tf.reduce_sum(mv_year_embedding, axis=-2)
        
        self.mv_year_model = tf.keras.Model(
            inputs=mv_year_input_layer, outputs=mv_year_embedding
        )
        
        # ====================================================
        # movie_genres
        # ====================================================
        mv_genre_input_layer = tf.keras.Input(
            name="movie_genres",
            shape=(self.max_genre_length,1),
            # shape=(1,),
            dtype=tf.string,
            # ragged=True
        )
        mv_genre_text = tf.keras.layers.TextVectorization(
            # max_tokens=max_tokens, 
            ngrams=2, 
            vocabulary=vocab_dict['movie_genre'],
            output_mode='int',
            output_sequence_length=self.max_genre_length,
        )(mv_genre_input_layer)
        mv_genre_embedding = tf.keras.layers.Embedding(
            # Let's use the explicit vocabulary lookup.
            input_dim=len(self.vocab_dict['movie_genre']) + self.num_oov_buckets,
            output_dim=self.mv_emb_size
        )(mv_genre_text)
        mv_genre_reshape = tf.keras.layers.Reshape([-1, self.mv_emb_size])(mv_genre_embedding)
        mv_g_avg_pooling = tf.keras.layers.GlobalAveragePooling1D()(mv_genre_reshape)
        self.mv_gen_model = tf.keras.Model(
            inputs=mv_genre_input_layer, outputs=mv_g_avg_pooling
        )
        
        # mv_genre_embedding = tf.reduce_sum(mv_genre_embedding, axis=-2)
        # mv_genre_pooling = tf.reduce_mean(mv_genre_embedding, axis=[0]) # axis=[0,1]
        
        # mv_genre_reshape = tf.expand_dims(mv_genre_pooling, axis=0)
        # self.mv_gen_model = tf.keras.Model(
        #     inputs=mv_genre_input_layer, outputs=mv_genre_reshape
        # )
        
        # ====================================================
        # movie_tags
        # ====================================================
        # mv_tags_input_layer = tf.keras.Input(
        #     name="movie_tags",
        #     shape=(TAG_MAX_LENGTH,1),
        #     # shape=(1,),
        #     dtype=tf.string,
        #     # ragged=True
        # )
        # mv_tags_text = tf.keras.layers.TextVectorization(
        #     # max_tokens=max_tokens, 
        #     ngrams=2, 
        #     vocabulary=vocab_dict['movie_tags'],
        #     output_mode='int',
        #     output_sequence_length=TAG_MAX_LENGTH,
        # )(mv_tags_input_layer)
        # mv_tags_embedding = tf.keras.layers.Embedding(
        #     # Let's use the explicit vocabulary lookup.
        #     input_dim=len(self.vocab_dict['movie_tags']) + self.num_oov_buckets,
        #     output_dim=self.mv_emb_size
        # )(mv_tags_text)
        # # mv_tags_pooling = tf.reduce_mean(mv_tags_embedding, axis=[0]) # axis=[0,1]
        # mv_tags_reshape = tf.keras.layers.Reshape([-1, self.mv_emb_size])(mv_tags_embedding)
        # mv_avg_pooling = tf.keras.layers.GlobalAveragePooling1D()(mv_tags_reshape)
        # self.mv_tags_model = tf.keras.Model(
        #     inputs=mv_tags_input_layer, outputs=mv_avg_pooling
        # )
        
        # mv_tags_reshape = tf.expand_dims(mv_tags_pooling, axis=0)
        # self.mv_tags_model = tf.keras.Model(
        #     inputs=mv_tags_input_layer, outputs=mv_tags_reshape
        # )
        
        # mv_tags_embedding = tf.reduce_sum(mv_tags_embedding, axis=-2)
        # tf.keras.layers.Reshape([-1, MAX_PLAYLIST_LENGTH, embedding_dim]),
        # mv_tags_pooling = tf.keras.layers.GlobalAveragePooling1D()(mv_tags_embedding)
        
        # ====================================================
        #
        # get global context (user) sampling function
        #
        # ====================================================
        
    # def _get_global_context_features(self, x) -> np.ndarray:
    def _get_global_context_features(self, x) -> tf.Tensor:
        """
        This function generates a single global observation vector.
        """

        _id = self.user_id_model(x['user_id'])
        _age = self.user_age_model(x['user_age']) # bucketized_user_age
        _occ = self.user_occ_model(x['user_occupation_text'])
        _ts = self.user_ts_model(x['target_rating_timestamp'])
        _ug = self.user_gender_model(x['user_gender'])
        _uz = self.user_zip_model(x['user_zip_code'])

        concat = tf.concat(
            [_id, _age, _occ, _ts, _ug, _uz], axis=-1
        )

        return concat
        
    # def _get_per_arm_features(self, x) -> np.ndarray:
    def _get_per_arm_features(self, x) -> tf.Tensor:
        """
        This function generates a single global observation vector.
        """
        _mid = self.mv_id_model(x['target_movie_id'])
        _myr = self.mv_year_model(x['target_movie_year'])
        _mtl = self.mv_title_model(x['target_movie_title'])

        BATCH_SIZE_g = x["target_movie_genres"].shape[0]
        GEN_LENGTH = x["target_movie_genres"].shape[1]
        _mgen = self.mv_gen_model(tf.reshape(x['target_movie_genres'], [BATCH_SIZE_g, GEN_LENGTH, 1]))

        # BATCH_SIZE_t = x["movie_tags"].shape[0]
        # TAG_LENGTH = x["movie_tags"].shape[1]
        # _mtag = self.mv_tags_model(tf.reshape(x['movie_tags'], [BATCH_SIZE_t, TAG_LENGTH, 1]))
        
        concat = tf.concat(
            [_mid, _mgen, _myr, _mtl], axis=-1 #_mtag
        )

        return concat