"""Model Factory for generating Models that use TF-Agents."""
import json
import os
import sys
import time
import random
import string
import argparse
import functools
from typing import List, Union
from pprint import pprint
import pickle as pkl
import numpy as np

# TF-Agent agents & networks
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.agents import neural_linucb_agent
from tf_agents.bandits.agents import neural_epsilon_greedy_agent
from tf_agents.bandits.networks import global_and_arm_feature_network
from tf_agents.bandits.policies import policy_utilities

from src.per_arm_rl import data_utils
from src.per_arm_rl import train_utils
from src.per_arm_rl import data_config

# logging
import logging
# logging.disable(logging.WARNING)

# tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# ====================================================
# get global context (user) feature embedding models 
# ====================================================

def get_user_id_emb_model(vocab_dict, num_oov_buckets, global_emb_size):
    
    user_id_input_layer = tf.keras.Input(
        name="user_id",
        shape=(1,),
        dtype=tf.string
    )

    user_id_lookup = tf.keras.layers.StringLookup(
        max_tokens=len(vocab_dict['user_id']) + num_oov_buckets,
        num_oov_indices=num_oov_buckets,
        mask_token=None,
        vocabulary=vocab_dict['user_id'],
    )(user_id_input_layer)

    user_id_embedding = tf.keras.layers.Embedding(
        # Let's use the explicit vocabulary lookup.
        input_dim=len(vocab_dict['user_id']) + num_oov_buckets,
        output_dim=global_emb_size
    )(user_id_lookup)
    
    user_id_embedding = tf.reduce_sum(user_id_embedding, axis=-2)
    user_id_model = tf.keras.Model(inputs=user_id_input_layer, outputs=user_id_embedding)
    
    return user_id_model

def get_user_age_emb_model(vocab_dict, num_oov_buckets, global_emb_size):
    
    user_age_input_layer = tf.keras.Input(
        name="bucketized_user_age",
        shape=(1,),
        dtype=tf.float32
    )

    user_age_lookup = tf.keras.layers.IntegerLookup(
        vocabulary=vocab_dict['bucketized_user_age'],
        num_oov_indices=num_oov_buckets,
        oov_value=0,
    )(user_age_input_layer)

    user_age_embedding = tf.keras.layers.Embedding(
        # Let's use the explicit vocabulary lookup.
        input_dim=len(vocab_dict['bucketized_user_age']) + num_oov_buckets,
        output_dim=global_emb_size
    )(user_age_lookup)

    user_age_embedding = tf.reduce_sum(user_age_embedding, axis=-2)
    user_age_model = tf.keras.Model(inputs=user_age_input_layer, outputs=user_age_embedding)
    
    return user_age_model

def get_user_occ_emb_model(vocab_dict, num_oov_buckets, global_emb_size):
    
    user_occ_input_layer = tf.keras.Input(
        name="user_occupation_text",
        shape=(1,),
        dtype=tf.string
    )
    user_occ_lookup = tf.keras.layers.StringLookup(
        max_tokens=len(vocab_dict['user_occupation_text']) + num_oov_buckets,
        num_oov_indices=num_oov_buckets,
        mask_token=None,
        vocabulary=vocab_dict['user_occupation_text'],
    )(user_occ_input_layer)
    
    user_occ_embedding = tf.keras.layers.Embedding(
        # Let's use the explicit vocabulary lookup.
        input_dim=len(vocab_dict['user_occupation_text']) + num_oov_buckets,
        output_dim=global_emb_size
    )(user_occ_lookup)
    
    user_occ_embedding = tf.reduce_sum(user_occ_embedding, axis=-2)
    user_occ_model = tf.keras.Model(inputs=user_occ_input_layer, outputs=user_occ_embedding)
    
    return user_occ_model

def get_ts_emb_model(vocab_dict, num_oov_buckets, global_emb_size):
    
    user_ts_input_layer = tf.keras.Input(
        name="timestamp",
        shape=(1,),
        dtype=tf.int64
    )

    user_ts_lookup = tf.keras.layers.Discretization(
        vocab_dict['timestamp_buckets'].tolist()
    )(user_ts_input_layer)

    user_ts_embedding = tf.keras.layers.Embedding(
        # Let's use the explicit vocabulary lookup.
        input_dim=len(vocab_dict['timestamp_buckets'].tolist()) + num_oov_buckets,
        output_dim=global_emb_size
    )(user_ts_lookup)

    user_ts_embedding = tf.reduce_sum(user_ts_embedding, axis=-2)
    user_ts_model = tf.keras.Model(inputs=user_ts_input_layer, outputs=user_ts_embedding)
    
    return user_ts_model

# ====================================================
# get perarm feature embedding models
# ====================================================

def get_mv_id_emb_model(vocab_dict, num_oov_buckets, mv_emb_size):
    
    mv_id_input_layer = tf.keras.Input(
        name="movie_id",
        shape=(1,),
        dtype=tf.string
    )

    mv_id_lookup = tf.keras.layers.StringLookup(
        max_tokens=len(vocab_dict['movie_id']) + num_oov_buckets,
        num_oov_indices=num_oov_buckets,
        mask_token=None,
        vocabulary=vocab_dict['movie_id'],
    )(mv_id_input_layer)

    mv_id_embedding = tf.keras.layers.Embedding(
        # Let's use the explicit vocabulary lookup.
        input_dim=len(vocab_dict['movie_id']) + num_oov_buckets,
        output_dim=mv_emb_size
    )(mv_id_lookup)

    mv_id_embedding = tf.reduce_sum(mv_id_embedding, axis=-2)
    mv_id_model = tf.keras.Model(inputs=mv_id_input_layer, outputs=mv_id_embedding)
    
    return mv_id_model

def get_mv_gen_emb_model(vocab_dict, num_oov_buckets, mv_emb_size):
    
    mv_genre_input_layer = tf.keras.Input(
        name="movie_genres",
        shape=(1,),
        dtype=tf.float32
    )

    mv_genre_lookup = tf.keras.layers.IntegerLookup(
        vocabulary=vocab_dict['movie_genres'],
        num_oov_indices=num_oov_buckets,
        oov_value=0,
    )(mv_genre_input_layer)

    mv_genre_embedding = tf.keras.layers.Embedding(
        # Let's use the explicit vocabulary lookup.
        input_dim=len(vocab_dict['movie_genres']) + num_oov_buckets,
        output_dim=mv_emb_size
    )(mv_genre_lookup)

    mv_genre_embedding = tf.reduce_sum(mv_genre_embedding, axis=-2)
    mv_gen_model = tf.keras.Model(inputs=mv_genre_input_layer, outputs=mv_genre_embedding)
    
    return mv_gen_model

# # ====================================================
# # get global_context_sampling_fn
# # ====================================================
# def _get_global_context_features(x):
#     """
#     This function generates a single global observation vector.
#     """
#     user_id_model = get_user_id_emb_model(
#         vocab_dict=VOCAB_DICT, 
#         num_oov_buckets=data_config.NUM_OOV_BUCKETS, 
#         global_emb_size=data_config.GLOBAL_EMBEDDING_SIZE
#     )
#     user_age_model = get_user_age_emb_model(
#         vocab_dict=VOCAB_DICT, 
#         num_oov_buckets=data_config.NUM_OOV_BUCKETS, 
#         global_emb_size=data_config.GLOBAL_EMBEDDING_SIZE
#     )
#     user_occ_model = get_user_occ_emb_model(
#         vocab_dict=VOCAB_DICT, 
#         num_oov_buckets=data_config.NUM_OOV_BUCKETS, 
#         global_emb_size=data_config.GLOBAL_EMBEDDING_SIZE
#     )
#     user_ts_model = get_ts_emb_model(
#         vocab_dict=VOCAB_DICT, 
#         num_oov_buckets=data_config.NUM_OOV_BUCKETS, 
#         global_emb_size=data_config.GLOBAL_EMBEDDING_SIZE
#     )

#     # for x in train_dataset.batch(1).take(1):
#     user_id_value = x['user_id']
#     user_age_value = x['bucketized_user_age']
#     user_occ_value = x['user_occupation_text']
#     user_ts_value = x['timestamp']

#     _id = user_id_model(user_id_value)
#     _age = user_age_model(user_age_value)
#     _occ = user_occ_model(user_occ_value)
#     _ts = user_ts_model(user_ts_value)

#     # to numpy array
#     _id = np.array(_id.numpy())
#     _age = np.array(_age.numpy())
#     _occ = np.array(_occ.numpy())
#     _ts = np.array(_ts.numpy())

#     concat = np.concatenate(
#         [_id, _age, _occ, _ts], axis=-1
#     ).astype(np.float32)

#     return concat

    
# # ====================================================
# # get per_arm_context_sampling_fn
# # ====================================================
# def _get_per_arm_features(x):
#     """
#     This function generates a single per-arm observation vector
#     """

#     mvid_model = get_mv_id_emb_model(
#         vocab_dict=VOCAB_DICT, 
#         num_oov_buckets=data_config.NUM_OOV_BUCKETS, 
#         mv_emb_size=data_config.MV_EMBEDDING_SIZE
#     )

#     mvgen_model = get_mv_gen_emb_model(
#         vocab_dict=VOCAB_DICT, 
#         num_oov_buckets=data_config.NUM_OOV_BUCKETS, 
#         mv_emb_size=data_config.MV_EMBEDDING_SIZE
#     )

#     # for x in train_dataset.batch(1).take(1):
#     mv_id_value = x['movie_id']
#     mv_gen_value = x['movie_genres'] #[0]

#     _mid = mvid_model(mv_id_value)
#     _mgen = mvgen_model(mv_gen_value)

#     # to numpy array
#     _mid = np.array(_mid.numpy())
#     _mgen = np.array(_mgen.numpy())

#     concat = np.concatenate(
#         [_mid, _mgen], axis=-1
#     ).astype(np.float32)

#     return concat

# ====================================================
# get agent
# ====================================================
def _get_agent(
    agent_type, 
    network_type, 
    time_step_spec, 
    action_spec, 
    observation_spec,
    global_step,
    global_layers,
    arm_layers,
    common_layers,
    agent_alpha,
    learning_rate,
    epsilon,
    encoding_dim,
    eps_phase_steps
):
    network = None
    observation_and_action_constraint_splitter = None
    PER_ARM = True

    if agent_type == 'LinUCB':
        agent = lin_ucb_agent.LinearUCBAgent(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            alpha=agent_alpha,
            accepts_per_arm_features=PER_ARM,
            dtype=tf.float32,
        )
    elif agent_type == 'LinTS':
        agent = lin_ts_agent.LinearThompsonSamplingAgent(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            alpha=agent_alpha,
            observation_and_action_constraint_splitter=(
                observation_and_action_constraint_splitter
            ),
            accepts_per_arm_features=PER_ARM,
            dtype=tf.float32,
        )
    elif agent_type == 'epsGreedy':
        # obs_spec = environment.observation_spec()
        if network_type == 'commontower':
            network = global_and_arm_feature_network.create_feed_forward_common_tower_network(
                observation_spec = observation_spec, 
                global_layers = global_layers, 
                arm_layers = arm_layers, 
                common_layers = common_layers,
                output_dim = 1
            )
        elif network_type == 'dotproduct':
            network = global_and_arm_feature_network.create_feed_forward_dot_product_network(
                observation_spec = observation_spec, 
                global_layers = global_layers, 
                arm_layers = arm_layers
            )
        agent = neural_epsilon_greedy_agent.NeuralEpsilonGreedyAgent(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            reward_network=network,
            optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=learning_rate
            ),
            epsilon=epsilon,
            observation_and_action_constraint_splitter=(
                observation_and_action_constraint_splitter
            ),
            accepts_per_arm_features=PER_ARM,
            emit_policy_info=(
                policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN,
                policy_utilities.BanditPolicyType.GREEDY,
            ),
            train_step_counter=global_step,
            name='OffpolicyNeuralEpsGreedyAgent'
        )

    elif agent_type == 'NeuralLinUCB':
        # obs_spec = environment.observation_spec()
        network = (
            global_and_arm_feature_network.create_feed_forward_common_tower_network(
                observation_spec = observation_spec, 
                global_layers = global_layers, 
                arm_layers = arm_layers, 
                common_layers = common_layers,
                output_dim = encoding_dim
            )
        )
        agent = neural_linucb_agent.NeuralLinUCBAgent(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            encoding_network=network,
            encoding_network_num_train_steps=eps_phase_steps,
            encoding_dim=encoding_dim,
            optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=learning_rate
            ),
            alpha=1.0,
            gamma=1.0,
            epsilon_greedy=epsilon,
            accepts_per_arm_features=PER_ARM,
            debug_summaries=True,
            summarize_grads_and_vars=True,
            emit_policy_info=policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN,
        )

    logging.info(f"Agent: {agent.name}\n")

    if network:
        logging.info(f"Network: {network.name}")
        network = network.name
    
    return agent, network