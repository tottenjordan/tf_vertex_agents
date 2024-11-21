"""
Top-K Off-Policy REINFORCE Agent

Top-K Off-Policy Correction for a REINFORCE Recommender System
Minmin Chen, Alex Beutel, Paul Covington, Sagar Jain, Francois Belletti, Ed Chi
https://arxiv.org/pdf/1812.02353.pdf

The main job of the Policy is to map observations from the user to actions. 
> the action is a set of K recommended items

To compute actions, the policy uses the network to compute the latent state s_t.
> The state s_t is multiplied by a trainable action embedding v_a for each action, 
  followed by a softmax to compute the action probabilities
  
Recompute latent states?
> when network weights are updated, should the latent state s_t be recomputed?
> maybe not necessary, since policy.action() supports obs with shape 
  either [B, ...] or [B, T, ...]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Callable, Optional, Sequence, Text, Tuple

# import gin
import tensorflow as tf
from tf_agents.networks import network
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common

IntegerLookup = tf.keras.layers.IntegerLookup

# Represents a function that takes in a state embedding and returns the scores
# of the closest actions and the actions themselves.
GetCandidateActionsFnType = Callable[
    [types.Float], Tuple[types.Float, types.Int]
]


#  This has to be called in a tf.function context for parallel_iterations to
#  have any effect.
def _deduplicate_actions(
    actions: types.Int, 
    num_select: int,
    parallel_iterations: int = 10
) -> types.Int:
    """
    Selects `num_select` unique actions from `actions`.

    Args:
        actions: A `[B, num_actions]` tensor of actions.
        num_select: Number of unique actions to select in each row.
        parallel_iterations: Number of parallel iterations for tf.map_fn.

    Returns:
        selected_actions: A `[B, num_select]` tensor of selected actions. Each
        row of `selected_actions` is chosen by taking unique values from the
        corresponding row of `actions`, padding with -1 if there are not enough
        unique values.
    """

    def select_unique_with_pad(vector):
        unique_values, _ = tf.unique(vector)
        pad_length = tf.maximum(num_select - tf.shape(unique_values)[0], 0)
        return tf.pad(unique_values, [[0, pad_length]], constant_values=-1)[
            :num_select
        ]

    return tf.map_fn(
        select_unique_with_pad, 
        actions, 
        parallel_iterations=parallel_iterations
    )


@common.function(autograph=True)
def _partial_greedy_sample(
    candidate_actions: types.Int,
    logits: types.Float,
    num_select: int,
    num_greedy_actions: int,
    oversample_multiplier: int,
) -> types.Int:
    """
    Performs partial greedy sampling of given actions.

    For each batch entry, this returns `num_select` actions. Out of this,
        `num_greedy_actions` are chosen greedily based on the given logits, and the
        rest are sampled with replacement from the remaining logits. The sampling
        with replacement is approximated by sampling extra actions (controlled by the
        `oversample_multiplier`) followed by deduplication.

    Args:
        candidate_actions: `[B, num_candidates]` int tensor of candidate actions to
           select from.
        logits: `[B, num_candidates]` float tensor of corresponding logits.
        num_select: Number of actions to return. A `[B, num_select]` tensor will be
           returned unless `num_select < num_candidates`, in which case all
           `candidate_actions` are returned.
        num_greedy_actions: Out of `num_select` actions in each row,
           `num_greedy_actions` actions will be selected greedily. The rest will be
           sampled from the remaining logits.
        oversample_multiplier: Amount of oversampling. Used to approximate sampling
           without replacement when choosing non-greedy actions.

    Returns:
        selected_actions: `[B, num_select]` tensor of actions selected
        from `candidate_actions`.
    """
    
    tf.debugging.assert_rank(candidate_actions, 2)
    tf.debugging.assert_rank(logits, 2)

    num_candidates = tf.shape(candidate_actions)[1]
    num_select = tf.minimum(num_select, num_candidates)
    num_greedy_actions = tf.minimum(num_greedy_actions, num_select)

    sorted_indices = tf.argsort(
        logits, axis=-1, direction='DESCENDING', stable=False
    )

    top_indices = sorted_indices[:, :num_greedy_actions]
    top_actions = tf.gather(candidate_actions, top_indices, batch_dims=1)

    num_to_sample = num_select - num_greedy_actions
  
    if num_to_sample == 0:
        return top_actions

    bottom_indices = sorted_indices[:, num_greedy_actions:]
    bottom_logits = tf.gather(logits, bottom_indices, batch_dims=1)
    bottom_actions = tf.gather(candidate_actions, bottom_indices, batch_dims=1)

    sampled_indices = tf.random.categorical(
        bottom_logits, 
        num_samples=num_to_sample * oversample_multiplier
    )
    sampled_actions = tf.gather(bottom_actions, sampled_indices, batch_dims=1)
    sampled_actions = _deduplicate_actions(sampled_actions, num_to_sample)

    return tf.concat([top_actions, sampled_actions], axis=1)


def _get_last_valid_states(states, step_types):
    """
    Returns the state corresponding to the last valid step.

    This function extracts the state corresponding to the last valid step. The
    `step_types` tensor is shaped `[B, T]`, corresponding to `B` trajectories with
    possibly different lengths.

    Trajectories shorter than `T` will be padded with multiple `StepType.LAST`, so
    in this case the last valid step is the first occurence of `StepType.LAST`.

    Trajectories longer than `T` will not have a `StepType.LAST`, so the last
    valid step is the final step at `T`.

    Once the last valid steps are calculated, the corresponding `states` are
    returned.

    Args:
    states: A `[B, T, ...]` tensor of states.
    step_types: A `[B, T]` tensor of step_types.

    Returns:
    last_states: A `[B, ...]` tensor of `states` (with time dimension removed)
      corresponding to the last valid states.
    """

    is_last = tf.cast(step_types == ts.StepType.LAST, dtype=tf.int32)
    is_last = tf.concat(
        [is_last[:, :-1], tf.expand_dims(tf.ones_like(is_last[:, -1]), -1)],
        axis=1,
    )
    indices = tf.argmax(is_last, axis=1)
  
    return tf.gather(states, indices, axis=1, batch_dims=1)


# @gin.configurable
class TopKOffPolicyReinforcePolicy(tf_policy.TFPolicy):
    """
    Class to build a policy for the Top-K Off Policy REINFORCE Agent.
    """

    def __init__(
        self,
        time_step_spec: ts.TimeStep,
        action_spec: types.NestedTensorSpec,
        state_embedding_network: network.Network,
        get_candidate_actions_fn: GetCandidateActionsFnType,
        inverse_action_lookup_layer: Optional[types.LookupLayer] = None,
        policy_state_spec: types.NestedTensorSpec = (),
        num_actions: int = 16,
        num_greedy_actions: int = 8,
        oversample_multiplier: int = 5,
        boltzmann_temperature: float = 1.0,
        emit_logits_as_info: bool = True,
        name: Optional[Text] = None,
    ):
        """
        Creates a policy for the Top-K Off Policy REINFORCE Agent.

        Args:
          time_step_spec: A `TimeStep` spec of the expected time_steps.
          action_spec: A nest of BoundedTensorSpec representing the actions. The
            actions are expected to be tf.int32 scalar indices.
          state_embedding_network: A tf_agents.network.Network to be used as
            call(observation, step_type, state) to compute the state embeddings.
            This is typically a RNN.
          get_candidate_actions_fn: A `callable(state_embedding) -> scores, actions`
            that returns the scores of the actions closest to the state embedding
            and the actions themselves. The returned `actions` is an int Tensor with
            shape `[B, num_candidate_actions]`, and `scores` is a float Tensor of
            the same shape. The scores are used as logits for computing action
            probabilities.
          inverse_action_lookup_layer: Optional Lookup layer to convert action
            indices to real actions. If None, `action()` returns action indices
            instead of real actions.
          policy_state_spec: A nest of TensorSpec representing the policy_state. If
            `None`, this is just the state spec of the state embedding network.
          num_actions: Number of actions recommended by the policy at a time. This
            is `k` in the paper. See Section 4.3.
          num_greedy_actions: Out of `num_actions` recommended by the policy,
            `num_greedy_actions` are chosen greedily based on the highest q value
            and the rest are drawn randomly based on the softmax.
          oversample_multiplier: Used to approximate sampling without replacement
            when selecting non greedy actions. We sample more actions than required
            with replacement and deduplicate. See `_partial_greedy_sample()` for
            more details.
          boltzmann_temperature: Boltzmann temperature used in the softmax policy.
          emit_logits_as_info: If True, PolicyStep.Info will contain logits. 
          name: The name of this agent. All variables in this module will fall under
            that name. Defaults to the class name.
        """
        
        state_embedding_network.create_variables(time_step_spec.observation)
        self._state_embedding_network = state_embedding_network

        if not policy_state_spec:
            policy_state_spec = state_embedding_network.state_spec

        self._get_candidate_actions_fn = get_candidate_actions_fn
        self._inverse_action_lookup_layer = inverse_action_lookup_layer
        self._num_actions = num_actions
        self._num_greedy_actions = num_greedy_actions
        self._oversample_multiplier = oversample_multiplier
        self._boltzmann_temperature = boltzmann_temperature
        self._emit_logits_as_info = emit_logits_as_info

        info_spec = (
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
            if emit_logits_as_info
            else ()
        )
        super(TopKOffPolicyReinforcePolicy, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy_state_spec=policy_state_spec,
            info_spec=info_spec,
            clip=False,
            # Shape of tensors in info is variable with ScaNN, so skip validation.
            validate_args=False,
            name=name,
        )

    def _variables(self) -> Sequence[tf.Variable]:
        return self._state_embedding_network.variables

    def _action(
        self, 
        time_step, 
        policy_state, 
        seed
    ) -> policy_step.PolicyStep:  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
        """
        Computes actions (items to recommend) based on observation history.

        Args:
          time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
            `time_step.observation` is expected to be a [B, T, ...] tensor history
            of interactions which involve past actions, rewards, context etc.
          policy_state: Tensor, or a nested dict, list or tuple of Tensors
            representing the previous policy_state. This is usually empty as the
            network state is recomputed from scratch every time.
          seed: seed for Boltzmann sampling.

        Returns:
          PolicyStep:  A `PolicyStep` named tuple. `PolicyStep.action` is a
          `[B, num_actions]` tensor of actions representing the items to recommend.
          If `emit_logits_as_info = True`, PolicyStep.info will be a tensor
          `logits` containing the logits of actions.
          
        # TODO: Handle this correctly when ScaNN is enabled. Need to detect when ScaNN used
        """
        state_embedding, policy_state = self._state_embedding_network(
            time_step.observation,
            step_type=time_step.step_type,
            network_state=policy_state,
        )

        state_embedding = _get_last_valid_states(
            state_embedding, time_step.step_type
        )

        logits, candidate_actions = self._get_candidate_actions_fn(state_embedding)
        logits = logits / self._boltzmann_temperature

        recommended_actions = _partial_greedy_sample(
            candidate_actions,
            logits,
            num_select=self._num_actions,
            num_greedy_actions=self._num_greedy_actions,
            oversample_multiplier=self._oversample_multiplier,
        )

        if self._inverse_action_lookup_layer is not None:
            recommended_actions = self._inverse_action_lookup_layer(
                recommended_actions
            )

        info = logits if self._emit_logits_as_info else ()
        
        return policy_step.PolicyStep(
            action=recommended_actions, 
            state=policy_state, 
            info=info
        )