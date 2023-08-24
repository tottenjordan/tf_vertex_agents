"""Class for creating models with TF-Agents"""

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

from typing import (
    Any, Iterable, List, Optional, 
    Type, Callable, Sequence, Text
)

# tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# TF-Agent agents & networks
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.agents import neural_linucb_agent
from tf_agents.bandits.agents import neural_epsilon_greedy_agent
from tf_agents.bandits.networks import global_and_arm_feature_network
from tf_agents.bandits.policies import policy_utilities

from tf_agents.typing import types

from src.per_arm_rl import data_utils
from src.per_arm_rl import train_utils
from src.per_arm_rl import data_config

# logging
import logging
logging.disable(logging.WARNING)


class PerArmAgentFactory:
    
    '''
    TODO: make network an attribute to print
    '''
    
    def __init__(
        self,
        agent_type: str,
        # network_type: str,
        # vocab_dict: dict,
        # num_oov_buckets: int,
        # global_emb_size: int,
        # mv_emb_size: int,
    ):
        
        self.agent_type = agent_type
        # self.network_type = network_type
        # self.vocab_dict = vocab_dict
        # self.num_oov_buckets = num_oov_buckets
        # self.global_emb_size = global_emb_size
        # self.mv_emb_size = mv_emb_size
        # self.global_layers = global_layers
        # self.arm_layers = arm_layers
        # self.common_layers = common_layers
        
    @classmethod
    def _get_agent(
        self, 
        agent_type: str,
        network_type: str,
        time_step_spec: types.TimeStep, 
        action_spec: types.BoundedTensorSpec,
        # optionals
        global_layers: Optional[Sequence[int]] = None,
        arm_layers: Optional[Sequence[int]] = None,
        common_layers: Optional[Sequence[int]] = None,
        observation_spec: Optional[types.NestedTensorSpec] = None,
        agent_alpha: Optional[float] = None,
        output_dim: Optional[int] = None,
        learning_rate: Optional[float] = None,
        epsilon: Optional[float] = None,
        train_step_counter: Optional[tf.Variable] = None,
        eps_phase_steps: Optional[int] = None,
        observation_and_action_constraint_splitter: Optional[types.Splitter] = None,
    ):
        """
        
        :param TODO:
        :return:
        """
        self.PER_ARM = True
        
        # self.index_name = index_name
        self.agent_type = agent_type
        self.network_type = network_type
        self.time_step_spec = time_step_spec
        self.action_spec = action_spec
        
        if global_layers is not None:
            self.global_layers = global_layers
            
        if arm_layers is not None:
            self.arm_layers = arm_layers
            
        if common_layers is not None:
            self.common_layers = common_layers
            
        if observation_spec is not None:
            self.observation_spec = observation_spec
            
        if agent_alpha is not None:
            self.agent_alpha = agent_alpha
            
        if output_dim is not None:
            self.output_dim = output_dim
            
        if learning_rate is not None:
            self.learning_rate = learning_rate
            
        if epsilon is not None:
            self.epsilon = epsilon
            
        if train_step_counter is not None:
            self.train_step_counter = train_step_counter
            
        if eps_phase_steps is not None:
            self.eps_phase_steps = eps_phase_steps
            
        if agent_type == 'LinUCB':
            agent = lin_ucb_agent.LinearUCBAgent(
                time_step_spec=self.time_step_spec,
                action_spec=self.action_spec,
                alpha=self.agent_alpha,
                accepts_per_arm_features=self.PER_ARM,
                dtype=tf.float32,
            )
        elif agent_type == 'LinTS':
            agent = lin_ts_agent.LinearThompsonSamplingAgent(
                time_step_spec=self.time_step_spec,
                action_spec=self.action_spec,
                alpha=self.agent_alpha,
                observation_and_action_constraint_splitter=(
                    observation_and_action_constraint_splitter
                ),
                accepts_per_arm_features=self.PER_ARM,
                dtype=tf.float32,
            )
        elif agent_type == 'epsGreedy':
            # obs_spec = environment.observation_spec()
            if network_type == 'commontower':
                network = global_and_arm_feature_network.create_feed_forward_common_tower_network(
                    observation_spec = self.observation_spec, 
                    global_layers = self.global_layers, 
                    arm_layers = self.arm_layers, 
                    common_layers = self.common_layers,
                    output_dim = self.output_dim
                )
            elif network_type == 'dotproduct':
                network = global_and_arm_feature_network.create_feed_forward_dot_product_network(
                    observation_spec = self.observation_spec, 
                    global_layers = self.global_layers, 
                    arm_layers = self.arm_layers
                )
            agent = neural_epsilon_greedy_agent.NeuralEpsilonGreedyAgent(
                time_step_spec=self.time_step_spec,
                action_spec=self.action_spec,
                reward_network=network,
                optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=self.learning_rate
                ),
                epsilon=self.epsilon,
                observation_and_action_constraint_splitter=(
                    observation_and_action_constraint_splitter
                ),
                accepts_per_arm_features=self.PER_ARM,
                emit_policy_info=(
                    policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN,
                    policy_utilities.BanditPolicyType.GREEDY,
                ),
                train_step_counter=train_step_counter,
                name='NeuralEpsGreedyAgent'
            )

        elif agent_type == 'NeuralLinUCB':
            # obs_spec = environment.observation_spec()
            network = (
                global_and_arm_feature_network.create_feed_forward_common_tower_network(
                    observation_spec = self.observation_spec, 
                    global_layers = self.global_layers, 
                    arm_layers = self.arm_layers, 
                    common_layers = self.common_layers,
                    output_dim = self.encoding_dim
                )
            )
            agent = neural_linucb_agent.NeuralLinUCBAgent(
                time_step_spec=self.time_step_spec,
                action_spec=self.action_spec,
                encoding_network=network,
                encoding_network_num_train_steps=self.eps_phase_steps,
                encoding_dim=self.encoding_dim,
                optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=self.learning_rate
                ),
                alpha=1.0,                                     # TODO - parameterize
                gamma=1.0,                                     # TODO - parameterize
                epsilon_greedy=self.epsilon,
                accepts_per_arm_features=self.PER_ARM,
                debug_summaries=True,                          # TODO - parameterize
                summarize_grads_and_vars=True,                 # TODO - parameterize
                emit_policy_info=(
                    policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN
                ),
            )
            
        return agent
        