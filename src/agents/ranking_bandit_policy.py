"""
Implements a Generalized Linear Ranking Bandit Policies of https://proceedings.mlr.press/v151/santara22a.html.

PS: Tested only with batch_size = 1
"""

from typing import Any, Optional, Sequence, Text

import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp

# from tf_agents.agents import tf_agent

from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.policies import tf_policy
from tf_agents.policies import utils as policy_utilities
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.typing import types

class FeedbackModel(object):
    """Enumeration of feedback models."""

    # No feedback model specified.
    UNKNOWN = 0
    # Cascading feedback model: A tuple of the chosen index and its value.
    CASCADING = 1
    # Score Vector feedback model: Every element in the output ranking receives a
    # score value.
    SCORE_VECTOR = 2


class ClickModel(object):
    """Enumeration of user click models."""

    # No feedback model specified.
    UNKNOWN = 0
    # For every dimension of the item space, a unit vector is added to the list of
    # available items. If one of these unit-vector items gets selected, it results
    # in a `no-click`.
    GHOST_ACTIONS = 1
    # Inner-product scores are calculated, and if none of the scores exceed a
    # given parameter, no item is clicked.
    DISTANCE_BASED = 2

# tfd = tfp.distributions

class GenLinearRankingBanditPolicy(tf_policy.TFPolicy):
    """Implements the 'Independent' ranking bandit policy."""

    def __init__(
        self,
        time_step_spec: types.TimeStep,
        action_spec: types.BoundedTensorSpec,
        inv_cov_matrix: types.Array,
        weight_vector: types.Array,
        retry_budget: types.Int = 1,
        retry_rewards: Optional[Sequence[types.Float]] = None,
        give_up_penalties: Optional[Sequence[types.Float]] = None,
        alpha: float = 1.0,
        add_bias: bool = False,
        emit_policy_info: Sequence[Text] = (),
        emit_log_probability: bool = False,
        accepts_per_arm_features: bool = True,
        observation_and_action_constraint_splitter: Optional[
           types.Splitter] = None,
        name: Optional[Text] = None
    ):
        
        self._inv_cov_matrix = inv_cov_matrix
        self._weight_vector = weight_vector
        self._retry_budget = retry_budget
        self._retry_rewards = (
            tf.ones(retry_budget, dtype=tf.float32) if retry_rewards is None else
            tf.constant(retry_rewards, dtype=tf.float32)
        )
        self._give_up_penalties = (
            tf.zeros(retry_budget +
                     1, dtype=tf.float32) if give_up_penalties is None else
            tf.constant(give_up_penalties, dtype=tf.float32)
        )
        assert tf.rank(self._retry_rewards) == 1, "Only 1D tensor is supported."
        assert tf.rank(self._give_up_penalties) == 1, "Only 1D tensor is supported."
        
        self._alpha = alpha
        self._add_bias = add_bias
        if tf.nest.is_nested(action_spec):
            raise ValueError("Nested `action_spec` is not supported.")
        self._num_actions = action_spec.maximum + 1
        self._give_up_action = tf.constant([action_spec.maximum])
        self._accepts_per_arm_features = accepts_per_arm_features

        self._check_input_variables()
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
        inv_cov_matrix_dim = tf.compat.dimension_value(
            tf.shape(self._inv_cov_matrix)[0])

        if self._overall_context_dim != inv_cov_matrix_dim:
            raise ValueError(
                "The dimension of matrix `inv_cov_matrix` must match "
                "overall context dimension {}. "
                "Got {} for `inv_cov_matrix`.".format(
                    self._overall_context_dim, inv_cov_matrix_dim
                )
            )

        weight_vector_dim = tf.compat.dimension_value(
            tf.shape(self._weight_vector)[0]
        )
        if self._overall_context_dim != weight_vector_dim:
            raise ValueError(
                "The dimension of vector `weight_vector` must match "
                "context  dimension {}. "
                "Got {} for `weight_vector`.".format(
                    self._overall_context_dim, weight_vector_dim)
            )

        self._dtype = self._weight_vector.dtype
        self._emit_policy_info = emit_policy_info
        self._emit_log_probability = emit_log_probability
        info_spec = self._populate_policy_info_spec(
            time_step_spec.observation, observation_and_action_constraint_splitter
        )

        self._best_retry_sequence = []

        super(
            GenLinearRankingBanditPolicy, self
        ).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            info_spec=info_spec,
            emit_log_probability=emit_log_probability,
            observation_and_action_constraint_splitter=(
                observation_and_action_constraint_splitter
            ),
            name=name
        )

    def _check_input_variables(self):
        assert self._accepts_per_arm_features, "Only supports per-arm features now."
        assert len(self._inv_cov_matrix.shape) == 2, (
            "`inv_cov_matrix` must be a dxd"
            " matrix but has shape"
            f"{self._inv_cov_matrix.shape}"
        )
        inv_cov_shape = self._inv_cov_matrix.shape
        assert inv_cov_shape[0] == inv_cov_shape[1], ("`inv_cov_matrix` must be a"
                                                      " square matrix.")
        assert len(self._weight_vector.shape) == 1, ("`weight_vector` must be of"
                                                     " dimension d but has shape "
                                                     f"{self._weight_vector.shape}."
                                                     )
        weight_shape = self._weight_vector.shape
        assert weight_shape[0] == inv_cov_shape[0], ("The first dimension of the "
                                                     "`inv_cov_matrix` and "
                                                     "`weight_vector` should match."
                                                     )

    def _variables(self):
        all_vars = [
            self._inv_cov_matrix,
            self._weight_vector
        ]
        return [
            v for v in tf.nest.flatten(all_vars) if isinstance(v, tf.Variable)
        ]

    def _populate_policy_info_spec(
        self, 
        observation_spec, 
        observation_and_action_constraint_splitter
    ):
        predicted_rewards_mean = ()
        if (policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN in
            self._emit_policy_info):
            predicted_rewards_mean = tensor_spec.TensorSpec(
                [self._num_actions],
                dtype=self._dtype
            )
        predicted_rewards_optimistic = ()
        if (policy_utilities.InfoFields.PREDICTED_REWARDS_OPTIMISTIC in 
            self._emit_policy_info):
            predicted_rewards_optimistic = tensor_spec.TensorSpec(
                [self._num_actions],
                dtype=self._dtype
            )
        predicted_rewards_sampled = ()
        if (policy_utilities.InfoFields.PREDICTED_REWARDS_SAMPLED in
            self._emit_policy_info):
            predicted_rewards_sampled = tensor_spec.TensorSpec(
                [self._num_actions],
                dtype=self._dtype
            )

        if self._accepts_per_arm_features:
            # The features for the chosen arm is saved to policy_info.
            chosen_arm_features_info = (
                policy_utilities.create_chosen_arm_features_info_spec(
                    observation_spec
                )
            )
            info_spec = policy_utilities.PerArmPolicyInfo(
                predicted_rewards_mean=predicted_rewards_mean,
                predicted_rewards_optimistic=predicted_rewards_optimistic,
                predicted_rewards_sampled=predicted_rewards_sampled,
                chosen_arm_features=chosen_arm_features_info
            )
        else:
            info_spec = policy_utilities.PolicyInfo(
                predicted_rewards_mean=predicted_rewards_mean,
                predicted_rewards_optimistic=predicted_rewards_optimistic,
                predicted_rewards_sampled=predicted_rewards_sampled
            )
        return info_spec

    def _get_current_observation(
        self, 
        global_observation, 
        arm_observations,
        arm_index
    ):
        """Helper function to construct the observation for a specific arm.

        This function constructs the observation depending if the policy accepts
        per-arm features or not. If not, it simply returns the original observation.
        If yes, it concatenates the global observation with the observation of the
        arm indexed by `arm_index`.

        Args:
          global_observation: A tensor of shape `[global_context_dim]`.
            The global part of the observation.
          arm_observations: A tensor of shape `[num_actions,
            arm_context_dim]`. The arm part of the observation, for all arms. If the
            policy does not accept per-arm features, this parameter is unused.
          arm_index: (int) The arm for which the observations to be returned.

        Returns:
          A tensor of shape  [global_context_dim+arm_context_dim] if arm features,
          [global_context_dim] otherwise.
        """
        if self._accepts_per_arm_features:
            current_arm = arm_observations[:, arm_index, :]
            current_observation = tf.concat(
                [global_observation, current_arm],
                axis=-1
            )
            return current_observation
        else:
            return global_observation

    def _split_observation(self, observation):
        """Splits the observation into global and arm observations."""
        if self._accepts_per_arm_features:
            global_observation = observation[bandit_spec_utils.GLOBAL_FEATURE_KEY]
            arm_observations = observation[bandit_spec_utils.PER_ARM_FEATURE_KEY]
            if not arm_observations.shape.is_compatible_with(
                [None, self._num_actions-1, self._arm_context_dim]
            ):
                # self._num_actions-1 because we do not need observation for the giveup
                # action.
                raise ValueError(
                    "Arm observation shape is expected to be {}. Got {}.".format(
                        [None, self._num_actions-1, self._arm_context_dim],
                        arm_observations.shape.as_list())
                )
        else:
            global_observation = observation
            arm_observations = None

        return global_observation, arm_observations

    def _estimate_rewards(self, time_step, policy_state) -> Any:
        observation = time_step.observation
        if self.observation_and_action_constraint_splitter is not None:
            observation, _ = self.observation_and_action_constraint_splitter(
                observation
            )
        observation = tf.nest.map_structure(
            lambda o: tf.cast(o, dtype=self._dtype), observation
        )
        global_observation, arm_observations = self._split_observation(observation)

        if self._add_bias:
            # The bias is added via a constant 1 feature.
            global_observation = tf.concat(
                [
                    global_observation,
                    tf.ones([tf.shape(global_observation)[0], 1], dtype=self._dtype)
                ], 
                axis=1
            )

        # Check the shape of the observation matrix. The observations can be
        # batched.
        if not global_observation.shape.is_compatible_with(
            [None, self._global_context_dim]):
            raise ValueError(
                "Global observation shape is expected to be {}. Got {}.".format(
                    [None, self._global_context_dim],
                    global_observation.shape.as_list()
                )
            )

        self._est_rewards = []
        confidence_intervals = []
        for k in range(self._num_actions - 1):
            # self._num_actions - 1 because the last action is the give up action that
            # does not need explicit ranking.
            current_observation = self._get_current_observation(
                global_observation, arm_observations, k)
            assert current_observation.shape.is_compatible_with(
                [1, self._overall_context_dim]
            )
            est_mean_reward = tf.reduce_sum(
                tf.squeeze(current_observation) * tf.squeeze(self._weight_vector)
            )
            self._est_rewards.append(
                tf.sigmoid(est_mean_reward)
            )

            ci = tf.matmul(
                tf.matmul(current_observation, self._inv_cov_matrix),
                current_observation, transpose_b=True
            )
            confidence_intervals.append(tf.squeeze(ci))

        optimistic_estimates = []
        for mean_reward, confidence in zip(
            self._est_rewards,
            confidence_intervals
        ):
            optimistic_estimates.append(
                tf.math.sigmoid(mean_reward) + self._alpha * tf.sqrt(confidence)
            )

        # Keeping the batch dimension during the squeeze, even if batch_size == 1.
        rewards_for_ranking = tf.squeeze(
            tf.stack(optimistic_estimates, axis=-1)
        )

        return rewards_for_ranking

    def _sort_arms_by_success_prob(self, arm_rewards) -> tf.Tensor:
        return tf.argsort(
            arm_rewards, 
            direction="DESCENDING"
        )

    def _evaluate_retry_seq(
        self, 
        retry_seq: types.Array,
        arm_success_probs: tf.Tensor
    ) -> types.Float:
        """Evaluate a retry seq.

        using a sequence of decreasing rewards and losses.

        Args:
          retry_seq: a sequence of actions to be evaluated as a retry sequence.
          arm_success_probs: arm success probs.

        Returns:
          value: evaluation of the retry sequence
        """

        if tf.equal(tf.size(retry_seq), 0):
            return self._give_up_penalties[0]

        value = tf.Variable(initial_value=0.0)

        for limit in range(len(retry_seq)):
            failure_part = retry_seq[:limit]
            joint_failure_prob = tf.math.reduce_prod(
                [
                    (1 - arm_success_probs[action]) for action in failure_part
                ]
            )
            success_candidate = retry_seq[limit]
            value = value + (
                joint_failure_prob * arm_success_probs[success_candidate] * self._retry_rewards[limit]
            )
        all_failure_joint_prob = tf.math.reduce_prod(
            [
                (1 - arm_success_probs[action]) for action in retry_seq
            ]
        )
        value = value + all_failure_joint_prob * self._give_up_penalties[
            len(retry_seq)
        ]

        return value

    def _chop_and_evaluate(
        self,
        arms_sorted_by_success_probs: Sequence[int],
        arm_success_probs: tf.Tensor
    ) -> Sequence[float]:
        evaluations = []
        for stop in range(
            min(len(arms_sorted_by_success_probs), self._retry_budget) + 1
        ):
            evaluations.append(
                self._evaluate_retry_seq(
                    arms_sorted_by_success_probs[:stop],
                    arm_success_probs
                )
            )
        return evaluations

    def _distribution(
        self, 
        time_step, 
        policy_state
    ) -> policy_step.PolicyStep:

        print(
            f"Distribution called with timestep: {time_step.step_type},"
            f"prev reward: {time_step.reward}"
        )
        observation = time_step.observation
        if self.observation_and_action_constraint_splitter is not None:
            observation, _ = self.observation_and_action_constraint_splitter(
                observation
            )
        observation = tf.nest.map_structure(
            lambda o: tf.cast(o, dtype=self._dtype),
            observation
        )
        _, arm_observations = self._split_observation(observation)

        if time_step.is_first():
            # Plan the entire retry sequence.
            self._rewards_for_ranking = self._estimate_rewards(
                time_step, policy_state
            )
            arm_indices_sorted_by_reward = self._sort_arms_by_success_prob(
                self._rewards_for_ranking
            )
            retry_seq_len_evaluations = self._chop_and_evaluate(
                arm_indices_sorted_by_reward, self._rewards_for_ranking
            )

            best_retry_seq_len = tf.argmax(retry_seq_len_evaluations, axis=0)

            self._best_retry_sequence = arm_indices_sorted_by_reward[:best_retry_seq_len]

            if len(self._best_retry_sequence) < self._retry_budget:
                # Add the give up action.
                tf.concat(
                    [
                        self._best_retry_sequence, 
                        tf.constant([self._num_actions])
                    ],
                    0
                )
            print("\n\n\n Best retry sequence PLANNED: ", self._best_retry_sequence)

        elif time_step.is_mid():
            # print("Best retry sequence now: ", self._best_retry_sequence)
              assert not tf.equal(tf.size(self._best_retry_sequence), 0), (
                  "self._best_retry_sequence should "
                  "not have been empty"
              )
        elif time_step.is_last():
            # Produce the give-up action only.
            self._best_retry_sequence = self._give_up_action

        # Yield the arms one by one.
        sequence_unstacked = tf.unstack(self._best_retry_sequence, axis=-1)
        chosen_action = tf.expand_dims(sequence_unstacked.pop(0), axis=0)
        print("Choosing action: ", chosen_action)

        if sequence_unstacked:
            self._best_retry_sequence = tf.stack(sequence_unstacked, axis=-1)
        else:
            self._best_retry_sequence = tf.constant([])

        action_distributions = tfp.distributions.Deterministic(loc=chosen_action)

        if tf.math.equal(chosen_action, self._give_up_action):
            batch_size, _, action_dim = tf.shape(arm_observations)
            chosen_action_feats = np.zeros([batch_size, action_dim], dtype=np.float32)
            chosen_action_feats[:, -1] = 1  # Assuming batch_size=1.
            policy_info = policy_utilities.PerArmPolicyInfo(
                predicted_rewards_optimistic=(),
                predicted_rewards_sampled=(),
                predicted_rewards_mean=(),
                chosen_arm_features=tf.constant(chosen_action_feats)
            )
        else:
            policy_info = policy_utilities.populate_policy_info(
                arm_observations, 
                chosen_action, 
                self._rewards_for_ranking,
                tf.stack(self._est_rewards, axis=-1), 
                self._emit_policy_info,
                self._accepts_per_arm_features
            )
        print("Chosen arm feats:", policy_info.chosen_arm_features)

        return policy_step.PolicyStep(
            action_distributions, 
            policy_state,
            policy_info
        )