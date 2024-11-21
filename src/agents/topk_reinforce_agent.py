"""
REINFORCE Recommender Agent

Top-K Off-Policy Correction for a REINFORCE Recommender System
Minmin Chen, Alex Beutel, Paul Covington, Sagar Jain, Francois Belletti, Ed Chi
https://arxiv.org/pdf/1812.02353.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
from typing import Optional, Text, Tuple

import gin
import tensorflow as tf
import tensorflow_recommenders as tfrs
from tf_agents.agents import data_converter
from tf_agents.agents import tf_agent
from tf_agents.networks import network
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory as traj
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import value_ops

# this repo
from . import topk_op_reinforce_policy as topk_op_reinforce_policy
from . import rfa_utils as rfa_utils

TOL = 1e-6

class GetCandidateActionsFn(tf.Module): # TODO: edit
    """
    A function used in the policy to retrieve the closest actions to a state.

    This has to be a class inheriting from tf.Module instead of a function, so
    that it can store the action embeddings and be serialized as a SavedModel
    with the policy.
    """

    def __init__(
        self,
        action_embeddings: types.Tensor,
        num_candidate_actions: Optional[int] = None,
    ):
        """
        Creates a GetCandidateActionsFn.
        
        # TODO: configure for Vertex Vector Search

        If `num_candidate_actions` is given, we create a ScaNN index of the action
        embeddings to quickly lookup the closest actions to a given state embedding.
        Otherwise, we return all actions and their scores (dot product to the state
        embedding).

        Args:
          action_embeddings: A [num_actions, embedding_size] float tensor of action
            embeddings.
          num_candidate_actions: Number of candidate actions to retrieve for each
            state.
        """
        self._action_embeddings = action_embeddings
        self._num_candidate_actions = num_candidate_actions

        if num_candidate_actions is not None:
            print("building scann index...")
            self._retrieval_model = tfrs.layers.factorized_top_k.ScaNN(
                k=num_candidate_actions,
                num_leaves=1000,
                num_leaves_to_search=100,
                # num_reordering_candidates=500,
            )
            self._retrieval_model.index(self._action_embeddings)
            self._get_candidate_actions_fn = self._retrieval_model
        else:
            self._retrieval_model = None

            def get_candidate_actions_fn(state_embeddings):
                scores = tf.matmul(
                    state_embeddings, 
                    self._action_embeddings, 
                    transpose_b=True
                )
                ids = tf.range(self._action_embeddings.shape[0])
                ids = tf.broadcast_to(ids, tf.shape(scores))
                return scores, ids

            self._get_candidate_actions_fn = get_candidate_actions_fn
            
    def __call__(
        self, 
        state_embeddings: types.Tensor
    ) -> Tuple[types.Tensor, types.Tensor]:
        """
        Retrieves the closest actions to the `state_embedding`.

        If ScaNN is used, the actions are retrieved in order of closeness. Otherwise
        all actions are returned.

        Args:
          state_embeddings: A [B, embedding_size] tensor of state embeddings.

        Returns:
          scores: [B, num_candidate_actions] float tensor of scores.
          action ids: [B, num_candidate_actions] int tensor of action ids.
        """
        return self._get_candidate_actions_fn(state_embeddings)
    
    def rebuild_index(self):
        """
        Rebuilds ScaNN index, e.g. after embeddings are updated by training.
        """
        if self._retrieval_model is not None:
            print("re-building scann index...")
            self._retrieval_model.index(self._action_embeddings)
            
class TopKOffPolicyReinforceAgentLoss( # TODO: edit
    collections.namedtuple(
        'TopKOffPolicyReinforceAgentLoss',
        ('corrected_reinforce_loss', 'behavior_loss', 'regularization_loss'),
    )
):
    """
    Losses for the TopKOffPolicyReinforceAgent.

    corrected_reinforce_loss: The corrected REINFORCE policy gradient loss.
    behavior_loss: The loss for training the estimate of the behavior policy.
    regularization_loss: Regularization loss of all the networks.
    """

    pass

class TopKOffPolicyReinforceAgent(tf_agent.TFAgent):
    """
    Off Policy Top-K REINFORCE Agent for Recommender Systems.
    https://arxiv.org/pdf/1812.02353.pdf
    """

    def __init__(
        self,
        time_step_spec: ts.TimeStep,
        action_spec: types.BoundedTensorSpec,
        state_embedding_network: network.Network,
        optimizer: types.Optimizer,
        off_policy_correction_exponent: Optional[int],
        policy_num_actions: int,
        num_candidate_actions: Optional[int],
        num_greedy_actions: int,
        sampled_softmax_num_negatives: Optional[int],
        gamma: float,
        action_lookup_layer: Optional[types.LookupLayer] = None,
        inverse_action_lookup_layer: Optional[types.LookupLayer] = None,
        behavior_loss_weight: float = 1.0,
        regularization_loss_weight: float = 1.0,
        oversample_multiplier: int = 5,
        boltzmann_temperature: float = 1.0,
        min_ratio_cap: float = 0.001,
        max_ratio_cap: float = 10.0,
        weights_ema_decay: float = 0.99,
        weights_ema_zero_debias: bool = False,
        gradient_clipping: Optional[types.Float] = None,
        use_supervised_loss_for_main_policy: bool = False,
        debug_summaries: bool = False,
        summarize_grads_and_vars: bool = False,
        train_step_counter: Optional[tf.Variable] = None,
        name: Optional[Text] = None,
    ):
        """
        Creates a TopK Off-Policy Reinforce Agent.
        
        Args:
          time_step_spec: A `TimeStep` spec of the expected time_steps.
          action_spec: A nest of BoundedTensorSpec representing the actions. The
            actions are expected to be tf.int32 scalar indices in [0, num_actions).
          state_embedding_network: A tf_agents.network.Network to be used as
            call(observation, step_type, state) to compute the state embeddings.
            This is typically a RNN.
          optimizer: Optimizer for training the networks.
          off_policy_correction_exponent: The K used in the off policy correction to
            compute alpha. See Section 4.3 of the paper. If None, no off policy
            correction is applied.
          policy_num_actions: Number of actions recommended by the policy at a time.
          num_candidate_actions: Number of actions to retrieve using SCANN in the
            policy. A softmax is computed on this reduced set instead of the whole
            vocabulary to improve efficiency. If set to None, all actions will be
            retrieved and used to compute the softmax.
          num_greedy_actions: Out of `num_actions` recommended by the policy,
            `num_greedy_actions` are chosen greedily based on the highest q value
            and the rest are drawn randomly based on the softmax.
          sampled_softmax_num_negatives: Number of `negative` actions used to
            compute the sampled_softmax loss. if None, regular softmax will be used
            instead of sampled_softmax.
          gamma: A discount factor for future rewards.
          action_lookup_layer: An Optional Lookup layer to convert real world
            actions to integer action indices. If not provided, actions provided to
            `train()` are assumed to be action indices instead of real actions. This
            can happen e.g. if the dataset is handling the look up of real world
            actions to action indices.
          inverse_action_lookup_layer: An Optional Lookup layer to map action
            indices back to real actions in `policy.action()`. If None, `action()`
            returns action indices and the caller is responsible for converting the
            indices to real actions.
          behavior_loss_weight: Scalar weight to be multiplied to the behavior loss.
          regularization_loss_weight: Scalar weight to be multiplied to the
            regularization loss.
          oversample_multiplier: Used in the policy to approximate sampling without
            replacement when selecting non greedy actions. We sample more actions
            than required with replacement and deduplicate.
          boltzmann_temperature: Boltzmann temperature used in the softmax policy.
          min_ratio_cap: Minimum value to cap the importance ratio in the off policy
            correction.
          max_ratio_cap: Maximum value to cap the importance ratio in the off policy
            correction.
          weights_ema_decay: The corrected reinforce loss is normalized by dividing
            by a moving average of the sum of weights. `weights_ema_decay` is the
            decay factor used to compute this moving average.
          weights_ema_zero_debias: Whether to use a zero debiasing correction while
            computing the moving average.
          gradient_clipping: Norm length to clip gradients.
          use_supervised_loss_for_main_policy: If True, trains the main policy using
            a supervised loss equal to the negative log probability of the actions,
            instead of the Off Policy REINFORCE loss. This is useful while
            debugging, e.g. as a sanity check that the model can mimick the dataset
            behavior.
          debug_summaries: A bool to gather debug summaries.
          summarize_grads_and_vars: If True, gradient and network variable summaries
            will be written during training.
          train_step_counter: An optional counter to increment every time the train
            op is run. Defaults to the global_step.
          name: The name of this agent. All variables in this module will fall under
            that name. Defaults to the class name.
            
        Raises:
          ValueError if:
            - Action indices do not start at 0, i.e. `action_spec.minimum` is not 0.
            - Or, the embedding produced by the state embedding network is not
                1 dimensional.
        """
        tf.Module.__init__(self, name=name)

        if action_spec.minimum != 0:
            raise ValueError(
                '`action_spec.minimum should be 0, but saw {}'.format(
                    action_spec.minimum
                )
            )
        self._vocabulary_size = action_spec.maximum + 1

        self._action_lookup_layer = action_lookup_layer
        self._inverse_action_lookup_layer = inverse_action_lookup_layer

        state_embedding_spec = state_embedding_network.create_variables(
            time_step_spec.observation
        )
        
        if len(state_embedding_spec.shape) != 1:
            raise ValueError(
                'Expected state embedding to be a vector. Received a '
                '`state_embedding_network` with inner output shape {}.'.format(
                    state_embedding_spec.shape
                )
            )
        self._state_embedding_network = state_embedding_network

        self._off_policy_correction_exponent = off_policy_correction_exponent
        self._embedding_size = state_embedding_spec.shape[0]
        self._create_action_embedding_layers()

        self._get_candidate_actions_fn = GetCandidateActionsFn(
            action_embeddings=self._policy_embedding_weights_layer.weights[0],
            num_candidate_actions=num_candidate_actions,
        )
        
        collect_policy = (
            topk_op_reinforce_policy.TopKOffPolicyReinforcePolicy(
                time_step_spec=time_step_spec,
                action_spec=action_spec,
                state_embedding_network=self._state_embedding_network,
                get_candidate_actions_fn=self._get_candidate_actions_fn,
                inverse_action_lookup_layer=self._inverse_action_lookup_layer,
                num_actions=policy_num_actions,
                num_greedy_actions=num_greedy_actions,
                oversample_multiplier=oversample_multiplier,
                boltzmann_temperature=boltzmann_temperature,
            )
        )

        policy = collect_policy
        
        self._behavior_loss_weight = behavior_loss_weight
        self._regularization_loss_weight = regularization_loss_weight

        self._policy_num_actions = policy_num_actions

        if sampled_softmax_num_negatives is not None:
            self._softmax_log_prob_fn = functools.partial(
                rfa_utils.sampled_softmax_log_prob,
                num_sampled=sampled_softmax_num_negatives,
                num_classes=self._vocabulary_size,
                outer_dims=2,
            )
        else:
            self._softmax_log_prob_fn = rfa_utils.softmax_log_prob

        self._min_ratio_cap = min_ratio_cap
        self._max_ratio_cap = max_ratio_cap

        self._optimizer = optimizer
        self._gamma = gamma
        self._gradient_clipping = gradient_clipping
        self._weights_sum_var = tf.Variable(
            initial_value=0, dtype=tf.float32, trainable=False
        )
        self._weights_ema = tf.train.ExponentialMovingAverage(
            weights_ema_decay, zero_debias=weights_ema_zero_debias
        )
        self._use_supervised_loss_for_main_policy = (
            use_supervised_loss_for_main_policy
        )

        training_data_spec = collect_policy.trajectory_spec.replace(policy_info=())

        super(TopKOffPolicyReinforceAgent, self).__init__(
            time_step_spec,
            action_spec,
            policy,
            collect_policy,
            train_sequence_length=None,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter,
            training_data_spec=training_data_spec,
        )

        self._as_trajectory = data_converter.AsTrajectory(self.data_context)
        
    def post_process_policy(
        self,
    ) -> topk_op_reinforce_policy.TopKOffPolicyReinforcePolicy:
        """
        Post process and return policy.

        If the agent has a SCANN retrieval model, this rebuilds the SCANN index
        after action embeddings are updated through training, and returns the post
        processed policy. Note that since the policy and collect_policy have a
        reference to the layer containing the index, these policies will be updated
        in place as well.

        Returns:
          The post processed policy.
        """
        self._get_candidate_actions_fn.rebuild_index()
        return self.policy
    
    def _create_action_embedding_layers(self):
        # This results in an error in train during apply_gradients:
        # "Cannot use a constraint function on a sparse variable."
        # embedding_constraint = tf.keras.constraints.UnitNorm(axis=1)
        embedding_constraint = None

        self._policy_embedding_weights_layer = tf.keras.layers.Embedding(
            self._vocabulary_size,
            self._embedding_size,
            embeddings_constraint=embedding_constraint,
        )
        self._policy_embedding_weights_layer.build([])

        self._policy_embedding_biases_layer = tf.keras.layers.Embedding(
            self._vocabulary_size, 1
        )
        self._policy_embedding_biases_layer.build([])

        if self._off_policy_correction_exponent is not None:
            self._behavior_embedding_weights_layer = tf.keras.layers.Embedding(
                self._vocabulary_size,
                self._embedding_size,
                embeddings_constraint=embedding_constraint,
            )
            self._behavior_embedding_weights_layer.build([])

            self._behavior_embedding_biases_layer = tf.keras.layers.Embedding(
                self._vocabulary_size, 1
            )
            self._behavior_embedding_biases_layer.build([])
            
    def _train(self, experience, weights):
        """
        Performs one train step and returns the loss.

        We allow `experience` to contain trajectories of different lengths in the
        time dimension, but these have to be padded with dummy values to have a
        constant size of `T` in the time dimension. 
        
        Both `trajectory.reward` and `weights` have to be 0 for these dummy values. 
        `experience` can be provided in other formats such as `Transition`'s if 
        they can be converted into Trajectories.

        Args:
          experience: A [B, T] tensor of trajectories. All experience along the time
            dimension is assumed to be from a single episode, therefore
            `experience.step_type` is ignored. Similarly `discount` is assumed to be
            all ones and ignored.
          weights: A [B, T] float tensor of weights. Each row of `weights` (along
            the time dimension) is usually a sequence of 0's, followed by a sequence
            of 1's, again followed by a sequence of 0's. This divides the trajectory
            into 3 parts. The first part is used to warm start the state embedding
            network. The second part part is used to compute losses. Returns are
            computed using the second and third parts. The length of these sections
            could be different for different batch items.

        Returns:
          loss: An instance of tf_agent.LossInfo containing the total and individual
            losses.
        """
        experience = self._as_trajectory(experience)
        
        with tf.GradientTape() as tape:
            loss_info = self.total_loss(experience, weights=weights, training=True)
            tf.debugging.check_numerics(loss_info.loss, 'Loss is inf or nan')
        
        variables_to_train = (
            self._state_embedding_network.trainable_weights
            + self._policy_embedding_weights_layer.trainable_weights
            + self._policy_embedding_biases_layer.trainable_weights
        )
        if self._off_policy_correction_exponent is not None:
            variables_to_train += (
                self._behavior_embedding_weights_layer.trainable_weights
                + self._behavior_embedding_biases_layer.trainable_weights
            )
            
        grads = tape.gradient(loss_info.loss, variables_to_train)
        
        grads_and_vars = list(zip(grads, variables_to_train))
        
        if self._gradient_clipping:
            grads_and_vars = eager_utils.clip_gradient_norms(
                grads_and_vars, self._gradient_clipping
            )

        if self._summarize_grads_and_vars:
            eager_utils.add_variables_summaries(
                grads_and_vars, self.train_step_counter
            )
            eager_utils.add_gradients_summaries(
                grads_and_vars, self.train_step_counter
            )

        self._optimizer.apply_gradients(grads_and_vars)
        
        if self.train_step_counter is not None:
            self.train_step_counter.assign_add(1)

        return loss_info
    
    # TODO: main policy and behavior policy to be trained on different data
    def total_loss(
        self,
        experience: traj.Trajectory,
        weights: types.Float,
        training: bool = False,
    ) -> tf_agent.LossInfo:
        """
        Computes the total loss.

        Args:
          experience: A [B, T] tensor of trajectories. All experience along the time
            dimension is assumed to be from a single episode, therefore
            `experience.step_type` is ignored. Similarly `discount` is assumed to be
            all ones and ignored.
          weights: A [B, T] float tensor of weights. Each row of `weights` (along
            the time dimension) is usually a sequence of 0's, followed by a sequence
            of 1's, again followed by a sequence of 0's. This divides the trajectory
            into 3 parts. The first part is used to warm start the state embedding
            network. The second part part is used to compute losses. Returns are
            computed using the second and third parts.
          training: Whether to use the network in `training` mode, e.g. if the
            network contains BatchNorm, DropOut layers etc.

        Returns:
          loss: An instance of tf_agent.LossInfo containing the total and individual
            losses.
        """
        actions = experience.action
        if self._action_lookup_layer is not None:
            actions = self._action_lookup_layer(actions)
        
        observations = experience.observation

        state_embedding, _ = self._state_embedding_network(
            observations,
            # step_type=step_types,
            network_state=(),
            training=training,
        )

        discounts = tf.ones_like(actions, dtype=tf.float32) * self._gamma
        returns = value_ops.discounted_return(
            experience.reward, discounts, time_major=False
        )
        returns = tf.stop_gradient(returns)

        summaries = {}
        summaries['actions'] = actions            # .numpy().astype('float64')
        summaries['observations'] = observations  # .numpy().astype('float64')
        summaries['rewards'] = experience.reward
        summaries['returns'] = returns
        summaries['state_embedding'] = state_embedding
        
        if self._debug_summaries:     
            with tf.name_scope('TotalLoss'):
                common.summarize_tensor_dict(summaries, self.train_step_counter)

        regularization_loss = (
            sum(self._state_embedding_network.losses)
            + sum(self._policy_embedding_weights_layer.losses)
            + sum(self._policy_embedding_biases_layer.losses)
        )

        if self._off_policy_correction_exponent is not None:
            behavior_action_log_prob = self._softmax_log_prob_fn(
                self._behavior_embedding_weights_layer.trainable_weights[0],
                tf.squeeze(
                    self._behavior_embedding_biases_layer.trainable_weights[0], axis=1
                ),
                classes=actions,
                inputs=tf.stop_gradient(state_embedding),
            )  # [B, T]
            behavioral_policy_loss = -tf.math.divide_no_nan(
                tf.reduce_sum(behavior_action_log_prob * weights),
                tf.reduce_sum(weights),
            )
            regularization_loss += sum(
                self._behavior_embedding_weights_layer.losses
            ) + sum(self._behavior_embedding_biases_layer.losses)
        else:
            behavior_action_log_prob = None
            behavioral_policy_loss = 0.0
            
        # Approx. log prob of chosen actions only. [B, T].
        policy_action_log_prob = self._softmax_log_prob_fn(
            self._policy_embedding_weights_layer.trainable_weights[0],
            tf.squeeze(
                self._policy_embedding_biases_layer.trainable_weights[0], axis=1
            ),
            classes=actions,
            inputs=state_embedding,
        )
        if self._use_supervised_loss_for_main_policy:
            main_policy_loss = -tf.math.divide_no_nan(
                tf.reduce_sum(policy_action_log_prob * weights),
                tf.reduce_sum(weights),
            )
        else:
            main_policy_loss = self._corrected_reinforce_loss(
                policy_action_log_prob, behavior_action_log_prob, returns, weights
            )
        
        # TODO: Use common.aggregate_losses() for distributed training
        # https://github.com/tensorflow/agents/blob/master/tf_agents/utils/common.py#L1401
        total_loss = (
            main_policy_loss
            + self._behavior_loss_weight * behavioral_policy_loss
            + self._regularization_loss_weight * regularization_loss
        )

        losses_dict = {
            'main_policy_loss': main_policy_loss,
            'behavioral_policy_loss': behavioral_policy_loss,
            'regularization_loss': regularization_loss,
        }

        loss_info_extra = TopKOffPolicyReinforceAgentLoss(
            corrected_reinforce_loss=main_policy_loss,
            behavior_loss=behavioral_policy_loss,
            regularization_loss=tf.constant(regularization_loss, dtype=tf.float32),
        )

        losses_dict['total_loss'] = total_loss  # Total loss not in loss_info_extra.
        common.summarize_scalar_dict(
            losses_dict, self.train_step_counter, name_scope='Losses/'
        )

        return tf_agent.LossInfo(total_loss, loss_info_extra)
    
    def _corrected_reinforce_loss(
        self,
        policy_action_log_prob: types.Float,
        behavior_action_log_prob: types.Float,
        returns: types.Float,
        weights: types.Float,
    ) -> types.Float:
        """
        Computes the corrected reinforce loss.

        Args:
          policy_action_log_prob: [B, T] tensor of log probabilities of the chosen
            actions according to the policy distribution.
          behavior_action_log_prob: [B, T] tensor of log probabilities of the same
            actions, but computed with the behavior distribution.
          returns: [B, T] tensor of returns.
          weights: [B, T] tensor of weights to apply to the per element loss.

        Returns:
          corrected_reinforce_loss: A scalar, the corrected reinforce loss.
        """
        summaries = {}
        summaries['policy_action_log_prob'] = policy_action_log_prob
        summaries['behavior_action_log_prob'] = behavior_action_log_prob
        summaries['returns'] = returns
        summaries['weights'] = weights

        if self._off_policy_correction_exponent is not None:
            action_prob = tf.math.exp(policy_action_log_prob)
            summaries['action_prob'] = action_prob

            alpha = 1.0 - tf.pow(
                (1.0 - action_prob), self._off_policy_correction_exponent
            )
            summaries['alpha'] = alpha

            beta = tf.math.exp(behavior_action_log_prob)
            summaries['beta'] = beta

            off_policy_correction = tf.stop_gradient(alpha / (beta + TOL))
            summaries['off_policy_correction'] = off_policy_correction

            off_policy_correction = tf.clip_by_value(
                off_policy_correction, self._min_ratio_cap, self._max_ratio_cap
            )
            summaries['off_policy_correction_clipped'] = off_policy_correction
        else:
            off_policy_correction = 1.0
            
        # log(alpha) is correct term mathematically, but can be too aggressive
        # log(action_prob) works better in practice
        corrected_reinforce_loss = (
            off_policy_correction * returns * policy_action_log_prob
        )
        summaries['corrected_reinforce_loss_unweighted'] = corrected_reinforce_loss

        corrected_reinforce_loss = -tf.reduce_sum(
            corrected_reinforce_loss * weights
        )

        # ema.apply only accepts TF2 variables, so first assign the sum 
        # to a variable, then apply variable to ema.
        self._weights_sum_var.assign(tf.reduce_sum(weights))
        self._weights_ema.apply([self._weights_sum_var])
        weights_ema_average = self._weights_ema.average(self._weights_sum_var)
        summaries['reinforce_loss_weights_ema'] = weights_ema_average

        if self._debug_summaries:
            with tf.name_scope('CorrectedReinforceLoss'):
                common.summarize_tensor_dict(summaries, self.train_step_counter)

        return tf.math.divide_no_nan(corrected_reinforce_loss, weights_ema_average)