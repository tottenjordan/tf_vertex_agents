"""Offline evaluation for a REINFORCE agent with Top-K off-policy correction."""

import collections
from typing import Dict, Optional, Sequence, Text

import tensorflow as tf
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common

from . import offline_metrics as offline_metrics

# helper fn: slice tensor
def get_slice_tensor_fn(i):
    
    def slice_tensor(tensor):
        return tensor[:, i, tf.newaxis]

    return slice_tensor

# evaluate fn
def evaluate(
    policy: tf_policy.TFPolicy,
    dataset: tf.data.Dataset,
    offline_eval_metrics: Sequence[offline_metrics.OfflineMetric],
    train_step: tf.Variable,
    summary_writer: tf.summary.SummaryWriter,
    summary_prefix: Optional[Text] = 'Metrics',
) -> Dict[Text, types.Tensor]:
    """
    Evaluate actions from the policy given a dataset of trajectories.

    Args:
      policy: A TFPolicy to compute actions.
      dataset: A TF Dataset returning (trajectories, weights), both [B, T]. The
          actions are predicted for each observation in the trajectory.
      offline_eval_metrics: List of offline evaluation metrics.
      train_step: The step at which we are evaluating the policy, saved to
          summaries.
      summary_writer: A SummaryWriter for generating TF summaries.
      summary_prefix: A prefix label.

    Returns:
      A dictionary of results {metric_name: metric_value}
    """
    for metric in offline_eval_metrics:
        metric.reset()
        
    for batch in dataset:
        trajectory, weights = batch
        valid_mask = (trajectory.step_type != ts.StepType.LAST) | (
            trajectory.next_step_type != ts.StepType.LAST
        )
        mask = tf.cast(valid_mask, tf.float32) * weights

        policy_state = policy.get_initial_state(trajectory.step_type.shape[0])
        # TODO: optimize by predicting actions only for valid steps.
        
        predicted_actions = []
        predicted_info = []

        for i in tf.range(trajectory.step_type.shape[1]):
            slice_tensor = get_slice_tensor_fn(i)
            observation = tf.nest.map_structure(slice_tensor, trajectory.observation)
            step_type = slice_tensor(trajectory.step_type)
            time_step = ts.TimeStep(
                step_type=step_type,
                observation=observation,
                reward=tf.zeros_like(step_type, tf.float32),
                discount=tf.ones_like(step_type, tf.float32),
            )
        
            action_step = policy.action(time_step, policy_state)
            policy_state = action_step.state
            predicted_actions.append(action_step.action)
            predicted_info.append(action_step.info)

        # If scann used, logits will not be computed for all actions. 
        # So in this case, either (1) do not emit logits in the policy 
        # or (2) emit a tuple (logits, canidate_actions), and update metrics 
        # such as WeightedReturn to use this structure
        predicted_policy_steps = policy_step.PolicyStep(
            action=tf.stack(predicted_actions, axis=1),
            info=tf.stack(predicted_info, axis=1),
            state=(),
        )

        for metric in offline_eval_metrics:
            metric(trajectory, predicted_policy_steps, mask)
        
    results = [(metric.name, metric.result()) for metric in offline_eval_metrics]
    
    if train_step is not None and summary_writer:
        with summary_writer.as_default():
            for name, value in results:
                tag = common.join_scope(summary_prefix, name)
                tf.summary.scalar(name=tag, data=value, step=train_step)

    return collections.OrderedDict(results)