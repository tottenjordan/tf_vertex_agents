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

from tf_agents.agents import tf_agent

# TF-Agent agents & networks
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.agents import neural_linucb_agent
from tf_agents.bandits.agents import neural_epsilon_greedy_agent
from tf_agents.bandits.networks import global_and_arm_feature_network
from tf_agents.bandits.policies import policy_utilities
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.agents import linear_thompson_sampling_agent as lin_ts_agent

# ranking agent
import gin
import enum
from tf_agents.agents import data_converter
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.utils import common
from tf_agents.utils import nest_utils

from tf_agents.typing import types
# import ranking_bandit_policy as ranking_bandit_policy
from . import ranking_bandit_policy as ranking_bandit_policy

# logging
import logging
logging.disable(logging.WARNING)

GLOBAL_FEATURE_KEY = bandit_spec_utils.GLOBAL_FEATURE_KEY
PER_ARM_FEATURE_KEY = bandit_spec_utils.PER_ARM_FEATURE_KEY
SINGLE_FEATURE_KEY = 'single'
REPEATED_FEATURE_KEY = 'repeated'


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
        summarize_grads_and_vars: Optional[bool] = False,
        debug_summaries: Optional[bool] = False
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
        self.summarize_grads_and_vars = summarize_grads_and_vars
        self.debug_summaries = debug_summaries
        
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
                summarize_grads_and_vars=self.summarize_grads_and_vars,
                enable_summaries=self.debug_summaries,
            )
        elif agent_type == 'LinTS':
            EMIT_POLICY_INFO = ('predicted_rewards_mean', 'bandit_policy_type')
            agent = lin_ts_agent.LinearThompsonSamplingAgent(
                time_step_spec=self.time_step_spec,
                action_spec=self.action_spec,
                alpha=self.agent_alpha,
                observation_and_action_constraint_splitter=(
                    observation_and_action_constraint_splitter
                ),
                accepts_per_arm_features=self.PER_ARM,
                dtype=tf.float32,
                summarize_grads_and_vars=self.summarize_grads_and_vars,
                enable_summaries=self.debug_summaries,
                emit_policy_info=EMIT_POLICY_INFO,
            )
        elif agent_type == 'epsGreedy':
            # obs_spec = environment.observation_spec()
            
            # The following defines what side information we want to get
            # as part of the policy info when we call policy network.
            EMIT_POLICY_INFO = ('predicted_rewards_mean', 'bandit_policy_type')
            
            # The following makes it so that the model always returns the
            # expected rewards even if the model is exploring. This means that
            # the largest predicted rewards may not match the selected action
            #  when the model is exploring (i.e. bandit_policy == UNIFORM == 2).
            # To find other available info fields see:
            #   /tf_agents/policies/utils.py
            GREEDY_INFO_FIELDS = ('predicted_rewards_mean',)
            
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
                summarize_grads_and_vars=self.summarize_grads_and_vars,
                enable_summaries=self.debug_summaries,
                emit_policy_info=EMIT_POLICY_INFO,
                # (
                    # 'predicted_rewards_mean', 'bandit_policy_type'      # <- use these
                    # policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN, # <- not these
                    # policy_utilities.BanditPolicyType.GREEDY,           # <- not these
                # ),
                train_step_counter=train_step_counter,
                info_fields_to_inherit_from_greedy=GREEDY_INFO_FIELDS, #('predicted_rewards_mean',),
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
                    output_dim = self.output_dim
                )
            )
            agent = neural_linucb_agent.NeuralLinUCBAgent(
                time_step_spec=self.time_step_spec,
                action_spec=self.action_spec,
                encoding_network=network,
                encoding_network_num_train_steps=self.eps_phase_steps,
                encoding_dim=self.output_dim,
                optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=self.learning_rate
                ),
                alpha=1.0,                                              # TODO - parameterize
                gamma=1.0,                                              # TODO - parameterize
                epsilon_greedy=self.epsilon,
                accepts_per_arm_features=self.PER_ARM,
                debug_summaries=self.debug_summaries,                   # TODO - parameterize
                summarize_grads_and_vars=self.summarize_grads_and_vars, # TODO - parameterize
                emit_policy_info=(
                    policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN
                ),
            )
            
        return agent

# ==============================================
# An agent that maintains linear 
# estimates for rewards and their uncertainty.
# ==============================================

class ExplorationPolicy(enum.Enum):
    """
    Possible exploration policies.
    """
    linear_ucb_policy = 1


class GenLinearRankingBanditVariableCollection(tf.Module):
    """
    A collection of variables used by `LinearBanditAgent`.
    """

    def __init__(
        self,
        context_dim: int,
        num_models: int,
        use_eigendecomp: bool = False,
        dtype: tf.DType = tf.float32,
        name: Optional[Text] = None
    ):
        """Initializes an instance of `GenLinearRankingBanditVariableCollection`.

        It creates all the variables needed for `GenLinearRankingBanditAgent`.

        Args:
          context_dim: (int) The context dimension of the bandit environment the
            agent will be used on.
          num_models: (int) The number of models maintained by the agent. This is
            either the same as the number of arms, or, if the agent accepts per-arm
            features, 1.  # TODO(santara) We should restrict ourselves to 1 for now
          use_eigendecomp: (bool) Whether the agent uses eigen decomposition for
            maintaining its internal state.
          dtype: The type of the variables. Should be one of `tf.float32` and
            `tf.float64`.
          name: (string) the name of this instance.
        """
        assert num_models == 1
        tf.Module.__init__(self, name=name)
        self.inv_cov_matrix = []
        self.weight_vector = []
        self.num_samples = []
        for k in range(num_models):
            self.inv_cov_matrix = tf.compat.v2.Variable(
                tf.eye(context_dim, context_dim, dtype=dtype),
                name='inv_cov_' + str(k)
            )
            self.weight_vector = tf.compat.v2.Variable(
                tf.zeros(context_dim, dtype=dtype), 
                name='weights_{}'.format(k)
            )
            self.num_samples = tf.compat.v2.Variable(
                tf.zeros([], dtype=dtype), 
                name='num_samples_{}'.format(k)
            )


def sherman_morrisson_update(
    inv_cov: types.Tensor, 
    obs: types.Tensor
) -> types.Tensor:
    
    """
    Implements the Sherman-Morrisson formula for inverse-covariance update.

    The Sherman Morrisson formula provides an efficient way to update the inverse
    of a covariance matrix when a new vectorized sample observation becomes
    available. Please refer to
    https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula for details.

    Args:
        inv_cov: Inverse Covariance matrix.
        obs: New observation for update.

    Returns:
        inv_cov: Updated Inverse Covariance matrix.
    """
    if tf.rank(obs) == 2:
        assert obs.shape[0] == 1, 'only batch_size=1 is supported at present.'
        obs = tf.squeeze(obs, axis=0)  # Remove the batch dim since batch_size=1.
    
    elif tf.rank(obs) > 2:
        raise ValueError(
            '`obs` must be of shape (1, ndim) or (ndim,). '
            f'Obtained: {obs.shape}'
        )
    
    obs = tf.expand_dims(obs, axis=1)
    
    v = tf.matmul(inv_cov, obs)
    
    inv_cov = inv_cov - (
        tf.matmul(v, v, transpose_b=True) /
        (1 + tf.matmul(obs, v, transpose_a=True))
    )
    
    return inv_cov


def weight_update(
    weight: types.Tensor, 
    inv_cov: types.Tensor,
    obs: types.Tensor, 
    success_flag: types.Float,
    learning_rate: types.Float
) -> types.Tensor:
    """
    Performs a Newton-style update step of the weights of the model.

    Applies the weight update rule described in step 5 of Algorithm 1 in
    go/bandits-with-retry-paper.

    Args:
        weight: The previous weight vector.
        inv_cov: Inverse covariance matrix after updating using the current
          observation.
        obs: Feature vector of the current arm.
        success_flag: The observed outcome for the current action: -1 for failure
          1 for success and 0 if unobserved.
        learning_rate: Learning rate for the update.

    Returns:
        weight: Updated weight vector.
    """
    if tf.rank(obs) == 2:
        assert obs.shape[0] == 1, 'only batch_size=1 is supported at present.'
        obs = tf.squeeze(obs, axis=0)  # Remove the batch dim since batch_size=1.
    elif tf.rank(obs) > 2:
        raise ValueError(
            '`obs` must be of shape (1, ndim) or (ndim,). '
            f'Obtained: {obs.shape}'
        )
    delta = tf.tensordot(obs, weight, 1)
    
    nabla = tf.sigmoid(-success_flag * delta) * success_flag * obs
    
    weight = learning_rate * tf.matmul(inv_cov, tf.expand_dims(nabla, axis=1))
    
    return tf.squeeze(weight, axis=1)


@gin.configurable
class GenLinearRankingBanditAgent(tf_agent.TFAgent):
    """An agent that maintains linear reward estimates and their uncertainties."""

    def __init__(
        self,
        exploration_policy: ExplorationPolicy,
        time_step_spec: types.TimeStep,
        action_spec: types.BoundedTensorSpec,
        retry_budget: types.Int = 1,
        retry_rewards: Optional[types.NestedArray] = None,
        give_up_penalties: Optional[types.NestedArray] = None,
        variable_collection: Optional[
            GenLinearRankingBanditVariableCollection] = None,
        alpha: float = 1.0,
        gamma: float = 1.0,
        use_eigendecomp: bool = False,
        tikhonov_weight: float = 1.0,
        add_bias: bool = False,
        emit_policy_info: Sequence[Text] = (),
        emit_log_probability: bool = False,
        observation_and_action_constraint_splitter: Optional[
            types.Splitter] = None,
        accepts_per_arm_features: bool = True,
        learning_rate: float = 0.1,
        debug_summaries: bool = False,
        summarize_grads_and_vars: bool = False,
        enable_summaries: bool = True,
        dtype: tf.DType = tf.float32,
        name: Optional[Text] = None
    ):
        """Initialize an instance of `LinearBanditAgent`.

        Args:
          exploration_policy: An Enum of type `ExplorationPolicy`. The kind of
            policy we use for exploration. Currently supported policies are
            `LinUCBPolicy` and `LinearThompsonSamplingPolicy`.
          time_step_spec: A `TimeStep` spec describing the expected `TimeStep`s.
          action_spec: A scalar `BoundedTensorSpec` with `int32` or `int64` dtype
            describing the number of actions for this agent.
          retry_budget: Maximum number of trials that the agent is allowed to make.
          retry_rewards: A list of retry success rewards of length (retry_budget).
            Each number must be in [0.0, 1.0].
          give_up_penalties: A list of give-up penalty rewards of length
            (retry_budget+1). Each number must be in [-1.0, 0.0].
          variable_collection: Instance of
            `GenLinearRankingBanditVariableCollection`. Collection of variables to
            be updated by the agent. If `None`, a new instance of
            `GenLinearRankingBanditVariableCollection` will be created.
          alpha: (float) positive scalar. This is the exploration parameter that
            multiplies the confidence intervals.
          gamma: a float forgetting factor in [0.0, 1.0]. When set to 1.0, the
            algorithm does not forget.
          use_eigendecomp: whether to use eigen-decomposition or not. The default
            solver is Conjugate Gradient.
          tikhonov_weight: (float) tikhonov regularization term.
          add_bias: If true, a bias term will be added to the linear reward
            estimation.
          emit_policy_info: (tuple of strings) what side information we want to get
            as part of the policy info. Allowed values can be found in
            `policy_utilities.PolicyInfo`.
          emit_log_probability: Whether the policy emits log-probabilities or not.
            Since the policy is deterministic, the probability is just 1.
          observation_and_action_constraint_splitter: A function used for masking
            valid/invalid actions with each state of the environment. The function
            takes in a full observation and returns a tuple consisting of 1) the
            part of the observation intended as input to the bandit agent and
            policy, and 2) the boolean mask. This function should also work with a
            `TensorSpec` as input, and should output `TensorSpec` objects for the
            observation and mask.
          accepts_per_arm_features: (bool) Whether the agent accepts per-arm
            features.
          learning_rate: A float denoting the learning rate to be used in the update
            of the weight vector.
          debug_summaries: A Python bool, default False. When True, debug summaries
            are gathered.
          summarize_grads_and_vars: A Python bool, default False. When True,
            gradients and network variable summaries are written during training.
          enable_summaries: A Python bool, default True. When False, all summaries
            (debug or otherwise) should not be written.
          dtype: The type of the parameters stored and updated by the agent. Should
            be one of `tf.float32` and `tf.float64`. Defaults to `tf.float32`.
          name: a name for this instance of `LinearBanditAgent`.
        """
        tf.Module.__init__(self, name=name)
        common.tf_agents_gauge.get_cell('TFABandit').set(True)
        self._num_actions = policy_utilities.get_num_actions_from_tensor_spec(
            action_spec
        )
        self._num_models = 1 if accepts_per_arm_features else self._num_actions
        self._observation_and_action_constraint_splitter = (
            observation_and_action_constraint_splitter
        )
        self._time_step_spec = time_step_spec
        
        if not accepts_per_arm_features:
            raise ValueError(
                'The current implementation only supports the case'
                'where we have per arm features.'
            )
        self._accepts_per_arm_features = accepts_per_arm_features
        self._add_bias = add_bias
        if observation_and_action_constraint_splitter is not None:
            context_spec, _ = observation_and_action_constraint_splitter(
                time_step_spec.observation
            )
        else:
            context_spec = time_step_spec.observation
        (self._global_context_dim,
         self._arm_context_dim) = bandit_spec_utils.get_context_dims_from_spec(
            context_spec, accepts_per_arm_features
        )
        if self._add_bias:
            # The bias is added via a constant 1 feature.
            self._global_context_dim += 1
        
        self._overall_context_dim = self._global_context_dim + self._arm_context_dim

        self._alpha = alpha
        
        if variable_collection is None:
            variable_collection = GenLinearRankingBanditVariableCollection(
                context_dim=self._overall_context_dim,
                num_models=self._num_models,
                use_eigendecomp=use_eigendecomp,
                dtype=dtype
            )
        elif not isinstance(
            variable_collection,
            GenLinearRankingBanditVariableCollection
        ):
            raise TypeError(
                'Parameter `variable_collection` should be '
                'of type `GenLinearRankingBanditVariableCollection`.'
            )
        self._variable_collection = variable_collection
        self._inv_cov_matrix = variable_collection.inv_cov_matrix
        self._weight_vector = variable_collection.weight_vector
        self._num_samples = variable_collection.num_samples

        self._learning_rate = learning_rate
        
        self._gamma = gamma
        if self._gamma < 0.0 or self._gamma > 1.0:
            raise ValueError('Forgetting factor `gamma` must be in [0.0, 1.0].')
        self._dtype = dtype

        policy = ranking_bandit_policy.GenLinearRankingBanditPolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            inv_cov_matrix=self._inv_cov_matrix,
            weight_vector=self._weight_vector,
            retry_budget=retry_budget,
            retry_rewards=retry_rewards,
            give_up_penalties=give_up_penalties,
            alpha=alpha,
            add_bias=add_bias,
            emit_policy_info=emit_policy_info,
            emit_log_probability=emit_log_probability,
            accepts_per_arm_features=accepts_per_arm_features,
            observation_and_action_constraint_splitter=(
                observation_and_action_constraint_splitter),
            name='Policy'
        )

        training_data_spec = None
        if accepts_per_arm_features:
            training_data_spec = bandit_spec_utils.drop_arm_observation(
                policy.trajectory_spec
            )
        
        super(GenLinearRankingBanditAgent, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy=policy,
            collect_policy=policy,
            training_data_spec=training_data_spec,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            enable_summaries=enable_summaries,
            train_sequence_length=None
        )
        
        self._as_trajectory = data_converter.AsTrajectory(
            self.data_context, sequence_length=None
        )
        self._cumulative_reward = tf.Variable(
            0.0, dtype=time_step_spec.reward.dtype
        )

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def inv_cov_matrix(self):
        return self._inv_cov_matrix

    @property
    def weight_vector(self):
        return self._weight_vector

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def alpha(self):
        return self._alpha

    def update_alpha(self, alpha):
        return tf.compat.v1.assign(self._alpha, alpha)

    def _initialize(self):
        tf.compat.v1.variables_initializer(self.variables)

    def compute_summaries(self, reward: types.Tensor):
        self._cumulative_reward.assign_add(tf.squeeze(reward))
        if self.summaries_enabled:
            with tf.name_scope('Returns/'):
                tf.compat.v2.summary.scalar(
                    name='cumulative_reward',
                    data=self._cumulative_reward,
                    step=self.train_step_counter
                )

            if self._summarize_grads_and_vars:
                with tf.name_scope('Variables/'):
                    for var in self.policy.variables():
                        var_name = var.name.replace(':', '_')
                        tf.compat.v2.summary.histogram(
                            name=var_name,
                            data=var,
                            step=self.train_step_counter
                        )
                        tf.compat.v2.summary.scalar(
                            name=var_name + '_value_norm',
                            data=tf.linalg.global_norm([var]),
                            step=self.train_step_counter
                        )

    def _process_experience(self, experience):
        """
        Given an experience, returns reward, action, observation, and batch size.

        Args:
          experience: An instance of trajectory. Every element in the trajectory has
          two batch dimensions.

        Returns:
          A tuple of reward, action, observation, and batch_size. All the outputs
            (except `batch_size`) have a single batch dimension of value
            `batch_size`.
        """

        if self._accepts_per_arm_features:
            return self._process_experience_per_arm(experience)
        else:
            return self._process_experience_global(experience)

    def _process_experience_per_arm(self, experience):
        """
        Processes the experience in case the agent accepts per-arm features.

        In the experience coming from the replay buffer, the reward (and all other
        elements) have two batch dimensions `batch_size` and `time_steps`, where
        `time_steps` is the number of driver steps executed in each training loop.
        We flatten the tensors in order to reflect the effective batch size. Then,
        all the necessary processing on the observation is done, including splitting
        the action mask if it is present.

        After the preprocessing, the per-arm part of the observation is copied over
        from the respective policy info field and concatenated with the global
        observation. The action tensor will be replaced by zeros, since in the
        per-arm case, there is only one reward model to update.

        Args:
          experience: An instance of trajectory. Every element in the trajectory has
          two batch dimensions.

        Returns:
          A tuple of reward, action, observation, and batch_size. All the outputs
            (except `batch_size`) have a single batch dimension of value
            `batch_size`.
        """
        reward, _ = nest_utils.flatten_multi_batched_nested_tensors(
            experience.reward, self._time_step_spec.reward
        )
        observation, _ = nest_utils.flatten_multi_batched_nested_tensors(
            experience.observation, self.training_data_spec.observation
        )

        if self._observation_and_action_constraint_splitter is not None:
            observation, _ = self._observation_and_action_constraint_splitter(
                observation
            )
        batch_size = 1
        global_observation = observation[bandit_spec_utils.GLOBAL_FEATURE_KEY]
        
        if self._add_bias:
            # The bias is added via a constant 1 feature.
            global_observation = tf.concat(
                [
                    global_observation,
                    tf.ones([batch_size, 1], dtype=global_observation.dtype)
                ],
                axis=1
            )

        # The arm observation we train on needs to be copied from the respective
        # policy info field to the per arm observation field. Pretending there was
        # only one action, we fill the action field with zeros.
        action = tf.zeros(shape=[batch_size], dtype=tf.int64)
        chosen_action, _ = nest_utils.flatten_multi_batched_nested_tensors(
            experience.policy_info.chosen_arm_features,
            self.policy.info_spec.chosen_arm_features
        )
        arm_observation = chosen_action
        overall_observation = tf.concat(
            [
                global_observation, arm_observation
            ], 
            axis=1
        )
        reward = tf.cast(reward, self._dtype)

        return reward, action, overall_observation, batch_size

    def _process_experience_global(self, experience):
        """Processes the experience in case the agent accepts only global features.

        In the experience coming from the replay buffer, the reward (and all other
        elements) have two batch dimensions `batch_size` and `time_steps`, where
        `time_steps` is the number of driver steps executed in each training loop.
        We flatten the tensors in order to reflect the effective batch size. Then,
        all the necessary processing on the observation is done, including splitting
        the action mask if it is present.

        Args:
          experience: An instance of trajectory. Every element in the trajectory has
          two batch dimensions.

        Returns:
          A tuple of reward, action, observation, and batch_size. All the outputs
            (except `batch_size`) have a single batch dimension of value
            `batch_size`.
        """
        reward, _ = nest_utils.flatten_multi_batched_nested_tensors(
            experience.reward, self._time_step_spec.reward
        )
        observation, _ = nest_utils.flatten_multi_batched_nested_tensors(
            experience.observation, self.training_data_spec.observation
        )
        action, _ = nest_utils.flatten_multi_batched_nested_tensors(
            experience.action, self._action_spec
        )
        batch_size = tf.cast(
            tf.compat.dimension_value(tf.shape(reward)[0]), dtype=tf.int64
        )

        if self._observation_and_action_constraint_splitter is not None:
            observation, _ = self._observation_and_action_constraint_splitter(
                observation
            )
        if self._add_bias:
            # The bias is added via a constant 1 feature.
            observation = tf.concat(
                [
                    observation, tf.ones([batch_size, 1], dtype=observation.dtype)
                ],
                axis=1
            )

        observation = tf.reshape(
            tf.cast(observation, self._dtype), [batch_size, -1]
        )
        reward = tf.cast(reward, self._dtype)

        return reward, action, observation, batch_size

    def _train(self, experience, weights=None):
        """
        Updates the policy based on the data in `experience`.

        Note that `experience` should only contain data points that this agent has
        not previously seen. If `experience` comes from a replay buffer, this buffer
        should be cleared between each call to `train`.

        Args:
          experience: A batch of experience data in the form of a `Trajectory`.
          weights: Unused.

        Returns:
            A `LossInfo` containing the loss *before* the training step is taken.
            In most cases, if `weights` is provided, the entries of this tuple will
            have been calculated with the weights.  Note that each Agent chooses
            its own method of applying weights.
        """
        experience = self._as_trajectory(experience)

        del weights  # unused
        reward, _, observation, batch_size = self._process_experience(
            experience
        )
        reward = tf.squeeze(reward)  # Since we only support batch_size=1 now.
        print(f'Reward observed: {reward}')

        success_flag = 2 * tf.cast(reward > 0, tf.float32) - 1

        # Update the inverse covariance matrix.
        for i in range(reward.shape[0]):
            if observation[i][-1].numpy() == 1.0:
                break
            cov_inv_new = sherman_morrisson_update(
                self._inv_cov_matrix,
                success_flag[i]*observation[i]
            )
            tf.compat.v1.assign(self._inv_cov_matrix, cov_inv_new)

            # Update the weight vector.
            weight_new = weight_update(
                self._weight_vector,
                cov_inv_new,
                observation[i],
                success_flag[i],
                self._learning_rate
            )
            tf.compat.v1.assign(self._weight_vector, weight_new)

        self.compute_summaries(tf.reduce_sum(reward))

        self._train_step_counter.assign_add(batch_size)

        return tf_agent.LossInfo(loss=(-self._cumulative_reward), extra=())