"""Class implementation of the per-arm Cascading Bandit environment for RecSys.
"""
import random
from typing import Optional, Text

import gin
import numpy as np
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

# from google3.learning.smartchoices.training.models.ranking.environments import cascading_bandit_py_environment
# from google3.learning.smartchoices.training.models.ranking.environments import dataset_utils

SUPPORTED_DATASETS = [
    'movielens',
]

GLOBAL_KEY = bandit_spec_utils.GLOBAL_FEATURE_KEY
PER_ARM_KEY = bandit_spec_utils.PER_ARM_FEATURE_KEY


@gin.configurable
class CascadingBanditPerArmPyEnvironmentForRecommendation(
    cascading_bandit_py_environment.CascadingBanditPyEnvironment):
  """Implements the per-arm version of Cascading Bandit environment for RecSys.

  This environment takes a recommendation dataset and converts it into a
  cascading bandit environment. We explain the required structure of the input
  dataset with the example of MovieLens-100 movie recommendation dataset
  available at:
  https://www.kaggle.com/prajitdatta/movielens-100k-dataset

  This dataset contains 100K ratings from 943 users on 1682 items.
  This csv list of:
  user id | item id | rating | timestamp.
  This environment computes a low-rank matrix factorization (using SVD) of the
  data matrix `A`, such that: `A ~= U * Sigma * V^T`.

  The environment uses the rows of `U` as global (or user) features, and the
  rows of `V` as per-arm (or item) features.

  The reward of recommending item `v` to user `u` is `u * Sigma * v^T`.
  """

  def __init__(self,
               dataset_name: Text,
               data_dir: Text,
               rank_k: int,
               budget: int,
               give_up_action: types.Int = -1,
               binary_reward: bool = True,
               batch_size: int = 1,
               num_actions: int = 50,
               csv_delimiter: Text = ',',
               name: Optional[Text] = 'movielens_per_arm'):
    """Initializes the Per-arm Cascading Bandit environment.

    Args:
      dataset_name: (string) Name of the recommendation system dataset.
      data_dir: (string) Directory where the data lies (in text form).
      rank_k : (int) Which rank to use in the matrix factorization. This will
        also be the feature dimension of both the user and the item features.
      budget: (int) Retry budget denoting how many times the agent is allowed to
        retry before the environment resets.
      give_up_action: (np.ndarray) The 'give-up' action.
      binary_reward: (bool) Whether the reward function is binary.
      batch_size: (int) Number of observations generated per call.
      num_actions: (int) How many items to choose from per round.
      csv_delimiter: (string) The delimiter to use in loading the data csv file.
      name: (string) The name of this environment instance.
    """
    assert dataset_name.lower() in SUPPORTED_DATASETS, (f'{dataset_name} not'
                                                        ' supported yet.')
    self._dataset_name = dataset_name.lower()
    assert batch_size == 1, 'Only batch_size = 1 is supported.'
    self._batch_size = batch_size
    assert num_actions >= rank_k, (f'At least {rank_k} actions are needed for '
                                   f'arm features of length {rank_k} but '
                                   f'only {num_actions} have been specified.')
    assert num_actions >= budget, (f'At least {rank_k} actions are needed for '
                                   f'an environment with retry budget of '
                                   f'{budget} but only {num_actions} have been '
                                   'specified.')
    self._context_dim = rank_k
    self._num_actions = num_actions
    self._binary_reward = binary_reward

    # Compute the matrix factorization.
    if 'movielens' in self._dataset_name:
      self._data_matrix = dataset_utils.load_movielens_data(
          data_dir, delimiter=csv_delimiter)
    else:
      raise NotImplementedError(
          f'Please implement a data loader for {self._dataset_name}')

    self._num_users, self._num_items = self._data_matrix.shape

    # Compute the SVD.
    u, s, vh = np.linalg.svd(self._data_matrix, full_matrices=False)

    # Keep only the largest singular values.
    self._u_hat = u[:, :rank_k].astype(np.float32)
    self._s_hat = s[:rank_k].astype(np.float32)
    self._v_hat = np.transpose(vh[:rank_k]).astype(np.float32)

    self._approx_ratings_matrix = np.matmul(self._u_hat * self._s_hat,
                                            np.transpose(self._v_hat))

    self._action_spec = array_spec.BoundedArraySpec(
        shape=(),
        dtype=np.int32,
        minimum=0,
        maximum=num_actions,  # Room for an extra 'give-up' action.
        name='action')
    observation_spec = {
        GLOBAL_KEY:
            array_spec.ArraySpec(shape=[rank_k], dtype=np.float32),
        PER_ARM_KEY:
            array_spec.ArraySpec(
                shape=[num_actions, rank_k+1], dtype=np.float32),
    }
    self._time_step_spec = ts.time_step_spec(observation_spec)

    self._current_user_index = 0
    self._previous_user_index = 0

    self._current_item_indices = np.zeros([num_actions], dtype=np.int32)
    self._previous_item_indices = np.zeros([num_actions], dtype=np.int32)

    self._observations = {
        GLOBAL_KEY:
            np.zeros([batch_size, rank_k]),
        PER_ARM_KEY:
            np.zeros([batch_size, num_actions, rank_k+1]),
    }

    super(CascadingBanditPerArmPyEnvironmentForRecommendation, self).__init__(
        budget, observation_spec, self._action_spec,
        give_up_action=give_up_action, batch_size=batch_size, name=name)

  @property
  def batched(self):
    return True

  def _step(self, action: types.NestedArray) -> ts.TimeStep:
    """Returns a time step containing the reward for the action taken.

    The returning time step also contains the next observation.
    It should not be overridden by bandit environment implementations.

    Args:
      action: The action taken by the Bandit policy.

    Returns:
      A time step containing the reward for the action taken and the next
      observation.
    """
    # This step will take an action and return a reward.
    action = np.asarray(action)
    assert action.shape[0] == self._batch_size, ('`action` must be an array of'
                                                 ' length = batch_size.')
    reward = self._apply_action(action)
    if self._binary_reward and reward[0] > 0:
      # Note that the test for success (reward > 0) only holds for
      # self._batch_size == 1 since we make that assumption at the moment.
      return ts.termination(self._observe(), reward,
                            outer_dims=self._batch_size)
    else:
      return super(CascadingBanditPerArmPyEnvironmentForRecommendation,
                   self)._step(action)

  def _observe(self, new_context: bool = False) -> types.NestedArray:
    if new_context:
      sampled_user_index = np.random.randint(
          self._num_users)
      self._previous_user_index = self._current_user_index
      self._current_user_index = sampled_user_index

      sampled_item_indices = random.sample(
          range(self._num_items), self._num_actions)

      flat_item_list = self._v_hat[sampled_item_indices]
      current_items = flat_item_list.reshape(
          [self._batch_size, self._num_actions, self._context_dim])
      current_items = np.concatenate(
          [current_items, np.zeros([self._batch_size, self._num_actions, 1],
                                   dtype=np.float32)],
          axis=-1)

      self._previous_item_indices = self._current_item_indices
      self._current_item_indices = sampled_item_indices

      self._observations = {
          GLOBAL_KEY:
              self._u_hat[sampled_user_index][np.newaxis, :],
          PER_ARM_KEY:
              current_items,
      }
    return self._observations

  def _apply_action(self, action: types.NestedArray) -> types.NestedArray:
    action = action[0]  # Since we only support batch_size=1 at the moment.
    if action == self._give_up_action:
      return np.asarray([0.0], dtype=np.float32)
    chosen_arm_index = self._current_item_indices[action]
    float_reward = self._approx_ratings_matrix[self._current_user_index,
                                               chosen_arm_index]
    if self._binary_reward:
      return np.asarray([(float_reward > 0.5)], dtype=np.float32)
    else:
      return np.asarray([float_reward], dtype=np.float32)

  def _rewards_for_all_actions(self):
    rewards_matrix = self._approx_ratings_matrix[
        self._previous_user_index, self._previous_item_indices]
    return rewards_matrix

  def compute_optimal_action(self):
    return np.argmax(self._rewards_for_all_actions(), axis=-1)

  def compute_optimal_reward(self):
    return np.max(self._rewards_for_all_actions(), axis=-1)