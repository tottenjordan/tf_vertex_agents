"""Metrics for offline evaluation given a trajectory and predicted actions."""

from typing import Callable, Optional, Text

import tensorflow as tf
from tf_agents.metrics import tf_metric
from tf_agents.trajectories import trajectory as traj
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import value_ops

ActionLookupType = Callable[[types.Int], types.Int]
FilterType = Callable[[types.Trajectory], types.Bool]

class OfflineMetric(tf_metric.TFStepMetric):
    """
    Base class for Offline metrics used with the TopK Off-Policy Agent.
    """

    def call(
        self,
        trajectory: traj.Trajectory,
        predicted_policy_step: types.PolicyStep,
        masks: Optional[types.Float] = None,
    ):
        """
        Updates the metric from a trajectory and predicted policy steps.

        Args:
          trajectory: A [B, T, ...] trajectory.
          predicted_policy_step: A PolicyStep tuple containing a [B, T, K] tensor of
            predicted actions.
          masks: A [B, T] float tensor of masks.
        """
        raise NotImplementedError()
        
class AccuracyAtK(OfflineMetric):
    """
    Measures the Accuracy at K.
    > TODO: change to Recall?

    Given a `trajectory`, where `trajectory.action` is a [B, T] tensor of observed
    actions, and a [B, T, K] `predicted_actions` tensor of K predictions per
    observed action, AccuracyAtK is the fraction of times where at least one of
    the K predicted actions matched the observed action.
    """

    def __init__(
        self,
        trajectory_filter: Optional[FilterType] = None,
        name: Text = 'AccuracyAtK',
        prefix: Text = 'Metrics',
        k: Optional[int] = None,
    ):
        """
        Creates an AccuracyAtK metric.

        Args:
          trajectory_filter: A callable that takes in a trajectory and returns a
            tensor mask of booleans. This metric will be computed only for
            time_steps where this mask is True.
          name: A string name of the metric.
          prefix: A string prefix to add to the metric for summaries.
          k: The number of predicted actions to use at each time step for computing
            accuracy. If the number of actions predicted by the policy is higher
            than k, only the first k actions are used. If the number of predicted
            actions is less than or equal to k, all predicted actions are used. Also
            if k is None, all predicted actions are used.
        """
        super(AccuracyAtK, self).__init__(name=name, prefix=prefix)
        self._trajectory_filter = trajectory_filter
        self._correct_predictions = common.create_variable(
            initial_value=0, dtype=tf.float32, shape=(), name='correct_predictions'
        )
        self._count = common.create_variable(
            initial_value=0, dtype=tf.float32, shape=(), name='count'
        )
        self._k = k
        
    def call(
        self,
        trajectory: traj.Trajectory,
        predicted_policy_step: types.PolicyStep,
        masks: Optional[types.Float] = None,
    ):
        """
        Updates the metric from a trajectory and predicted policy steps.

        Args:
          trajectory: A batched trajectory where each element is shaped [B, T, ...].
          predicted_policy_step: A PolicyStep tuple containing a [B, T, K] tensor of
            predicted actions.
          masks: A [B, T] shaped float tensor of masks.
        """
        if masks is None:
            masks = tf.ones_like(trajectory.action, dtype=tf.float32)

        if self._trajectory_filter is not None:
            traj_mask = tf.cast(self._trajectory_filter(trajectory), tf.float32)
            masks = masks * traj_mask

        observed_actions = tf.expand_dims(trajectory.action, axis=2)
        predicted_actions = predicted_policy_step.action

        if self._k is not None and self._k < predicted_actions.shape[2]:
            predicted_actions = predicted_actions[:, :, : self._k]

        correct_predictions = tf.reduce_any(
            predicted_actions == observed_actions, axis=2
        )
        correct_predictions = tf.cast(correct_predictions, dtype=tf.float32)
        self._correct_predictions.assign_add(
            tf.reduce_sum(correct_predictions * masks)
        )
        self._count.assign_add(tf.reduce_sum(masks))

    def result(self) -> types.Float:
        return tf.math.divide_no_nan(
            self._correct_predictions, 
            self._count
        )

    def reset(self):
        self._correct_predictions.assign(0)
        self._count.assign(0)
        
class AveragePerClassAccuracyAtK(OfflineMetric):
    """
    Measures the Accuracy at K.

    Given a `trajectory`, where `trajectory.action` is a [B, T] tensor of observed
    actions, and a [B, T, K] `predicted_actions` tensor of K predictions per
    observed action, `AccuracyAtK` is the fraction of times where at least one
    of the K predicted actions matched the observed action. This metric computes
    `AccuracyAtK` for each class (action) separately and averages the result
    across classes.
    """

    def __init__(
        self,
        vocabulary_size: int,
        action_lookup: Optional[ActionLookupType] = None,
        trajectory_filter: Optional[FilterType] = None,
        name: Text = 'AveragePerClassAccuracyAtK',
        prefix: Text = 'Metrics',
        k: Optional[int] = None,
    ):
        """
        Creates a AveragePerClassAccuracyAtK metric.

        Args:
          vocabulary_size: The number of unique actions.
          action_lookup: A callable for mapping real world actions in an arbitrary
            range to actions within the vocabulary [0, vocabulary_size).
          trajectory_filter: A callable that takes in a trajectory and returns a
            tensor mask of booleans. This metric will be computed only for time
            steps where this mask is True.
          name: A string name of the metric.
          prefix: A string prefix to add to the metric for summaries.
          k: The number of predicted actions to use at each time step for computing
            accuracy. If the number of actions predicted by the policy is higher
            than k, only the first k actions are used. If the number of predicted
            actions is less than or equal to k, all predicted actions are used. Also
            if k is None, all predicted actions are used.
        """
        
        super(AveragePerClassAccuracyAtK, self).__init__(name=name, prefix=prefix)
        self._trajectory_filter = trajectory_filter
        self._vocabulary_size = vocabulary_size
        self._action_lookup = action_lookup
        self._correct_predictions = common.create_variable(
            initial_value=0,
            dtype=tf.float32,
            shape=(self._vocabulary_size),
            name='correct_predictions',
        )
        self._counts = common.create_variable(
            initial_value=0,
            dtype=tf.int32,
            shape=(self._vocabulary_size),
            name='counts',
        )
        self._k = k
        
    def call(
        self,
        trajectory: traj.Trajectory,
        predicted_policy_step: types.PolicyStep,
        masks: Optional[types.Float] = None,
    ):
        """
        Updates the metric from a trajectory and predicted policy steps.

        Args:
          trajectory: A batched trajectory where each element is shaped [B, T, ...].
          predicted_policy_step: A PolicyStep tuple containing a [B, T, K] tensor of
            predicted actions.
          masks: A [B, T] shaped float tensor of masks.
        """
        observed_actions = trajectory.action
        predicted_actions = predicted_policy_step.action
        
        if self._action_lookup:
            predicted_actions = self._action_lookup(predicted_actions)
            observed_actions = self._action_lookup(observed_actions)

        if masks is None:
            masks = tf.ones_like(observed_actions, dtype=tf.float32)
        
        if self._trajectory_filter is not None:
            traj_mask = tf.cast(self._trajectory_filter(trajectory), tf.float32)
            masks = masks * traj_mask

        if self._k is not None and self._k < predicted_actions.shape[2]:
            predicted_actions = predicted_actions[:, :, : self._k]

        correct_predictions = predicted_actions == tf.expand_dims(
            observed_actions, axis=2
        )
        correct_predictions = tf.cast(
            tf.reduce_any(correct_predictions, axis=2), dtype=tf.float32
        )

        self._correct_predictions.scatter_add(
            tf.IndexedSlices(
                values=correct_predictions * masks, indices=observed_actions
            )
        )
        self._counts.scatter_add(
            tf.IndexedSlices(
                values=tf.cast(masks, tf.int32), indices=observed_actions
            )
        )
        
    def result(self) -> types.Float:
        """
        Computes average accuracy over actions we have seen so far
        """
        non_zero_counts = self._counts > 0
        
        if tf.reduce_any(non_zero_counts):
            return tf.reduce_mean(
                self._correct_predictions[non_zero_counts]
                / tf.cast(self._counts[non_zero_counts], tf.float32)
          )
        else:
            return 0.0

    def reset(self):
        self._correct_predictions.assign(self._correct_predictions * 0)
        self._counts.assign(self._counts * 0)
        
# ========================================
# JT HERE
# ========================================


class LastActionAccuracyAtK(OfflineMetric):
    """
    Measures the Accuracy in predicting the last action.

    Given a `trajectory`, where `trajectory.action` is a [B, T] tensor of observed
    actions, and a [B, T, K] `predicted_actions` tensor of K predictions per
    observed action, LastActionAccuracyAtK is the fraction of sequences where the
    action corresponding to the last non zero mask was predicted correctly.
    """
    
    def __init__(
        self,
        trajectory_filter: Optional[FilterType] = None,
        name: Text = 'LastActionAccuracyAtK',
        prefix: Text = 'Metrics',
        k: Optional[int] = None,
    ):
        """
        Creates a LastActionAccuracyAtK metric.

        Args:
          trajectory_filter: A callable that takes in a trajectory and returns a
            tensor mask of booleans. This metric will be computed only for
            time_steps where this mask is True.
          name: A string name of the metric.
          prefix: A string prefix to add to the metric for summaries.
          k: The number of predicted actions to use at each time step for computing
            accuracy. If the number of actions predicted by the policy is higher
            than k, only the first k actions are used. If the number of predicted
            actions is less than or equal to k, all predicted actions are used. Also
            if k is None, all predicted actions are used.
        """
        super(LastActionAccuracyAtK, self).__init__(name=name, prefix=prefix)
        self._trajectory_filter = trajectory_filter
        self._correct_predictions = common.create_variable(
            initial_value=0, dtype=tf.float32, shape=(), name='correct_predictions'
        )
        self._count = common.create_variable(
            initial_value=0, dtype=tf.float32, shape=(), name='count'
        )
        self._k = k

    def call(
        self,
        trajectory: traj.Trajectory,
        predicted_policy_step: types.PolicyStep,
        masks: Optional[types.Float] = None,
    ):
        """
        Updates the metric from a trajectory and predicted policy steps.

        Args:
          trajectory: A batched trajectory where each element is shaped [B, T, ...].
          predicted_policy_step: A PolicyStep tuple containing a [B, T, K] tensor of
            predicted actions.
          masks: A [B, T] shaped float tensor of masks.
        """
        if masks is None:
            masks = tf.ones_like(trajectory.action, dtype=tf.float32)

        if self._trajectory_filter is not None:
            traj_mask = tf.cast(self._trajectory_filter(trajectory), tf.float32)
            masks = masks * traj_mask

        # Given a [B, T] mask, for each row, we want to find a mask for the last
        # non zero entry. e.g. (B = 1):
        #                                              mask = 1 0 1 1 0 0 0
        #                                          1 - mask = 0 1 0 0 1 1 1
        # cumprod(1 - mask, reverse=True, exclusive = True) = 0 0 0 1 1 1 1
        #                                         last_mask = 0 0 0 1 0 0 0
        last_mask = masks * tf.math.cumprod(
            1.0 - masks, exclusive=True, reverse=True, axis=1
        )

        observed_actions = tf.expand_dims(trajectory.action, axis=2)
        predicted_actions = predicted_policy_step.action

        if self._k is not None and self._k < predicted_actions.shape[2]:
            predicted_actions = predicted_actions[:, :, : self._k]

        correct_predictions = tf.reduce_any(
            predicted_actions == observed_actions, axis=2
        )
        correct_predictions = tf.cast(correct_predictions, dtype=tf.float32)
        self._correct_predictions.assign_add(
            tf.reduce_sum(correct_predictions * last_mask)
        )

        batch_size = tf.shape(masks)[0]
        self._count.assign_add(tf.cast(batch_size, dtype=tf.float32))

    def result(self) -> types.Float:
        return tf.math.divide_no_nan(self._correct_predictions, self._count)

    def reset(self):
        self._correct_predictions.assign(0)
        self._count.assign(0)


class ReturnWeightedAccuracyAtK(OfflineMetric):
    """
    Measures the ReturnWeightedAccuracyAtK.

    This is similar to the `AccuracyAtK` metric, except that each correct
    prediction is weighted by the return (instead of 1 as in `AccuracyAtK`).
    """

    def __init__(
        self,
        name: Text = 'ReturnWeightedAccuracyAtK',
        prefix: Text = 'Metrics',
        gamma: float = 1.0,
    ):
        """
        Creates a AveragePerClassAccuracyAtK metric.

        Args:
          name: A string name of the metric.
          prefix: A string prefix to add to the metric for summaries.
          gamma: discount factor for the reward to compute returns.
        """
        super(ReturnWeightedAccuracyAtK, self).__init__(name=name, prefix=prefix)

        self._gamma = gamma
        self._sum_values = common.create_variable(
            initial_value=0, dtype=tf.float32, shape=(), name='sum_values'
        )
        self._count = common.create_variable(
            initial_value=0, dtype=tf.float32, shape=(), name='counts'
        )

    def call(
        self,
        trajectory: traj.Trajectory,
        predicted_policy_step: types.PolicyStep,
        masks: Optional[types.Float] = None,
    ):
        """
        Updates the metric from a trajectory and predicted policy steps.

        Args:
          trajectory: A batched trajectory where each element is shaped [B, T, ...].
          predicted_policy_step: A PolicyStep tuple containing a [B, T, K] tensor of
            predicted actions.
          masks: A [B, T] shaped float tensor of masks.
        """
        observed_actions = trajectory.action
        batch_size = tf.shape(observed_actions)[0]
        if masks is None:
            masks = tf.ones_like(observed_actions, dtype=tf.float32)

        discounts = trajectory.discount * self._gamma
        returns = value_ops.discounted_return(
            trajectory.reward, discounts, time_major=False
        )

        predicted_actions = predicted_policy_step.action
        correct_predictions = predicted_actions == tf.expand_dims(
            observed_actions, axis=2
        )
        correct_predictions = tf.reduce_any(correct_predictions, axis=2)
        correct_predictions = tf.cast(correct_predictions, dtype=tf.float32)

        return_weighted_accuracy = tf.reduce_sum(
            returns * correct_predictions * masks
        )

        self._sum_values.assign_add(return_weighted_accuracy)
        self._count.assign_add(tf.cast(batch_size, dtype=tf.float32))

    def result(self) -> types.Float:
        """Computes the metric over trajectories seen so far."""
        return tf.math.divide_no_nan(self._sum_values, self._count)

    def reset(self):
        self._sum_values.assign(0.0)
        self._count.assign(0.0)


class WeightedReturns(OfflineMetric):
    """
    Measures returns weighted by probability or log probability of actions.

    Weighting by log probability is the same as the Reinforce objective. Weighting
    by probabilities is also useful as it is less sensitive than log probabilities
    at values close to 0.
    """

    def __init__(
        self,
        name: Text = 'WeightedReturns',
        prefix: Text = 'Metrics',
        action_lookup: Optional[ActionLookupType] = None,
        weight_by_probabilities: bool = False,
        gamma: float = 1.0,
    ):
        """
        Creates a WeightedReturns metric.

        Args:
          name: A string name of the metric.
          prefix: A string prefix to add to the metric for summaries.
          action_lookup: A callable for mapping real world actions in an arbitrary
            range to actions within a vocabulary.
          weight_by_probabilities: If True, the returns are weighted by
            probabilities instead of log probabilities.
          gamma: discount factor for the reward to compute returns.
        """
        super(WeightedReturns, self).__init__(name=name, prefix=prefix)

        self._gamma = gamma
        self._action_lookup = action_lookup
        self._weight_by_probabilities = weight_by_probabilities
        self._sum_values = common.create_variable(
            initial_value=0, dtype=tf.float32, shape=(), name='sum_values'
        )
        self._count = common.create_variable(
            initial_value=0, dtype=tf.float32, shape=(), name='counts'
        )

    def call(
        self,
        trajectory: traj.Trajectory,
        predicted_policy_step: types.PolicyStep,
        masks: Optional[types.Float] = None,
    ):
        """
        Updates the metric from a trajectory and predicted policy steps.

        Args:
          trajectory: A batched trajectory where each element is shaped [B, T, ...].
          predicted_policy_step: A PolicyStep tuple whose info field contains a [B,
            T, vocabulary_size] tensor of logits for all actions.
          masks: A [B, T] shaped float tensor of masks.
        """
        observed_actions = trajectory.action
        batch_size = tf.shape(observed_actions)[0]
        logits = predicted_policy_step.info

        if masks is None:
            masks = tf.ones_like(observed_actions, dtype=tf.float32)

        if self._action_lookup:
            observed_actions = self._action_lookup(observed_actions)

        discounts = trajectory.discount * self._gamma
        returns = value_ops.discounted_return(
            trajectory.reward, discounts, time_major=False
        )

        # log probability of observed actions.
        weights = -tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=observed_actions, logits=logits
        )
        if self._weight_by_probabilities:
            weights = tf.math.exp(weights)

        weighted_returns = returns * weights * masks
        self._sum_values.assign_add(tf.reduce_sum(weighted_returns))
        self._count.assign_add(tf.cast(batch_size, dtype=tf.float32))

    def result(self) -> types.Float:
        """Computes the metric over trajectories seen so far."""
        return tf.math.divide_no_nan(self._sum_values, self._count)

    def reset(self):
        self._sum_values.assign(0.0)
        self._count.assign(0.0)