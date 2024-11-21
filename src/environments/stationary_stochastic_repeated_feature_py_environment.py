"""Stationary Stochastic Python Bandit environment with repeated features."""

from typing import Callable, Optional, Sequence, Text

import gin
import numpy as np
from tf_agents.bandits.environments import bandit_py_environment
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.specs import array_spec
from tf_agents.typing import types

GLOBAL_KEY = bandit_spec_utils.GLOBAL_FEATURE_KEY
PER_ARM_KEY = bandit_spec_utils.PER_ARM_FEATURE_KEY
NUM_ACTIONS_KEY = bandit_spec_utils.NUM_ACTIONS_FEATURE_KEY


# @gin.configurable
class StationaryStochasticRepeatedFeaturePyEnvironment(
    bandit_py_environment.BanditPyEnvironment
):
  """Stationary Stochastic Bandit environment with repeated features."""

  def __init__(
      self,
      global_single_context_sampling_fn: Callable[[], types.Array],
      arm_single_context_sampling_fn: Callable[[], types.Array],
      max_num_actions: int,
      global_repeated_context_sampling_fn: Callable[[], types.Array],
      arm_repeated_context_sampling_fn: Callable[[], types.Array],
      num_repetitions: int,
      reducer_fn: Callable[[types.Array], types.Array],
      reward_fn: Callable[[types.Array], Sequence[float]],
      num_actions_fn: Optional[Callable[[], int]] = None,
      batch_size: Optional[int] = 1,
      name: Optional[Text] = 'stationary_stochastic_per_arm',
  ):
    """Initializes the environment.

    In each round, single and repeated features for both global and per-arm
    part of the observation are generated. The reward_fn
    function takes the concatenation of the 4 parts, and outputs a possibly
    random reward.

    In case `num_action_fn` is specified, the number of actions will be dynamic
    and a `num_actions` feature key indicates the number of actions in any given
    sample.

    Example:
      def global_single_context_sampling_fn():
        # 2-dimensional global single features.
        return np.random.randint(0, 10, [2])

      def arm_single_context_sampling_fn():
        # 3-dimensional arm single features.
        return np.random.randint(-3, 4, [3])

      def global_repeated_context_sampling_fn():
        # 4-dimensional global repeated features.
        return np.random.randint(0, 10, [4])

      def arm_repeated_context_sampling_fn():
        # 5-dimensional arm single features.
        return np.random.randint(-3, 4, [5])

      def reducer_fn(tensor):
        # Averages the features along the repetition dimension.
        return np.mean(tensor, axis=-2)

      def reward_fn(x):
        return sum(x)

      def num_actions_fn():
        return np.random.randint(2, 6)

      env = StationaryStochasticRepeatedFeaturePyEnvironment(
          global_single_context_sampling_fn,
          arm_single_context_sampling_fn,
          4,
          global_repeated_context_sampling_fn,
          arm_repeated_context_sampling_fn,
          5,
          reward_fn,
          reducer_fn,
          num_actions_fn)

    Args:
      global_single_context_sampling_fn: A function that outputs a random 1d
        array or list of ints or floats. This output is the single global
        context. Its shape and type must be consistent across calls.
      arm_single_context_sampling_fn: A function that outputs a random 1d array
        or list of ints or floats (same type as the output of
        `global_context_sampling_fn`). This output is the sinlge per-arm
        context. Its shape must be consistent across calls.
      max_num_actions: (int) the maximum number of actions in every sample. If
        `num_actions_fn` is not set, this many actions are available in every
        time step.
      global_repeated_context_sampling_fn: A function that outputs a random 1d
        array or list of ints or floats. This output is the repeated global
        context. Its shape and type must be consistent across calls.
      arm_repeated_context_sampling_fn: A function that outputs a random 1d
        array or list of ints or floats (same type as the output of
        `global_context_sampling_fn`). This output is the repeated per-arm
        context. Its shape must be consistent across calls.
      num_repetitions: (int) the repetition number of the repeated features. The
        number of repetitions is constant accross time steps and the same for
        global and per-arm features.
      reducer_fn: A function that reduces the repeated features to a single
        array.
      reward_fn: A function that generates a reward when called with an
        observation.
      num_actions_fn: If set, it should be a function that outputs a single
        integer specifying the number of actions for a given time step. The
        value output by this function will be capped between 1 and
        `max_num_actions`. The number of actions will be encoded in the
        observation by the feature key `num_actions`.
      batch_size: The batch size.
      name: The name of this environment instance.
    """
    self._global_single_context_sampling_fn = global_single_context_sampling_fn
    self._global_repeated_context_sampling_fn = (
        global_repeated_context_sampling_fn
    )
    self._arm_single_context_sampling_fn = arm_single_context_sampling_fn
    self._arm_repeated_context_sampling_fn = arm_repeated_context_sampling_fn
    self._max_num_actions = max_num_actions
    self._num_repetitions = num_repetitions
    self._reducer_fn = reducer_fn
    self._reward_fn = reward_fn
    self._batch_size = batch_size
    self._num_actions_fn = num_actions_fn

    observation_spec = {
        GLOBAL_KEY: {
            'single': array_spec.ArraySpec.from_array(
                global_single_context_sampling_fn()
            ),
            'repeated': array_spec.add_outer_dims_nest(
                array_spec.ArraySpec.from_array(
                    global_repeated_context_sampling_fn()
                ),
                (num_repetitions,),
            ),
        },
        PER_ARM_KEY: {
            'single': array_spec.add_outer_dims_nest(
                array_spec.ArraySpec.from_array(
                    arm_single_context_sampling_fn()
                ),
                (max_num_actions,),
            ),
            'repeated': array_spec.add_outer_dims_nest(
                array_spec.ArraySpec.from_array(
                    arm_repeated_context_sampling_fn()
                ),
                (max_num_actions, num_repetitions),
            ),
        },
    }
    if self._num_actions_fn is not None:
      num_actions_spec = array_spec.BoundedArraySpec(
          shape=(),
          dtype=np.dtype(type(self._num_actions_fn())),
          minimum=1,
          maximum=max_num_actions,
      )
      observation_spec.update({NUM_ACTIONS_KEY: num_actions_spec})

    action_spec = array_spec.BoundedArraySpec(
        shape=(),
        dtype=np.int32,
        minimum=0,
        maximum=max_num_actions - 1,
        name='action',
    )

    super(StationaryStochasticRepeatedFeaturePyEnvironment, self).__init__(
        observation_spec, action_spec, name=name
    )

  def batched(self) -> bool:
    return True

  @property
  def batch_size(self) -> int:
    return self._batch_size

  def _observe(self) -> types.NestedArray:
    global_single_obs = np.stack(
        [
            self._global_single_context_sampling_fn()
            for _ in range(self._batch_size)
        ]
    )
    global_repeated_obs = np.reshape(
        [
            self._global_repeated_context_sampling_fn()
            for _ in range(self._batch_size * self._num_repetitions)
        ],
        (self._batch_size, self._num_repetitions, -1),
    )
    arm_single_obs = np.reshape(
        [
            self._arm_single_context_sampling_fn()
            for _ in range(self._batch_size * self._max_num_actions)
        ],
        (self._batch_size, self._max_num_actions, -1),
    )
    arm_repeated_obs = np.reshape(
        [
            self._arm_repeated_context_sampling_fn()
            for _ in range(
                self._batch_size * self._num_repetitions * self._max_num_actions
            )
        ],
        (self._batch_size, self._max_num_actions, self._num_repetitions, -1),
    )
    self._observation = {
        GLOBAL_KEY: {
            'single': global_single_obs,
            'repeated': global_repeated_obs,
        },
        PER_ARM_KEY: {'single': arm_single_obs, 'repeated': arm_repeated_obs},
    }

    if self._num_actions_fn:
      num_actions = [self._num_actions_fn() for _ in range(self._batch_size)]
      num_actions = np.maximum(num_actions, 1)
      num_actions = np.minimum(num_actions, self._max_num_actions)
      self._observation.update({NUM_ACTIONS_KEY: num_actions})
    return self._observation

  def _apply_action(self, action: np.ndarray) -> types.Array:
    if action.shape[0] != self.batch_size:
      raise ValueError('Number of actions must match batch size.')
    global_obs = self._observation[GLOBAL_KEY]
    global_single_obs = global_obs['single']
    global_repeated_obs = global_obs['repeated']
    batch_size_range = range(self.batch_size)
    reduced_global_repeated_obs = np.stack(
        [self._reducer_fn(global_repeated_obs[i]) for i in batch_size_range]
    )
    arm_single_obs = self._observation[PER_ARM_KEY]['single'][
        batch_size_range, action, :
    ]
    arm_repeated_obs = self._observation[PER_ARM_KEY]['repeated'][
        batch_size_range, action, :, :
    ]
    reduced_arm_repeated_obs = np.stack(
        [self._reducer_fn(arm_repeated_obs[i]) for i in batch_size_range]
    )

    def concatenate_features(ind: int) -> np.array:
      """Concatenates all needed features for a single element in the batch."""
      return np.concatenate((
          global_single_obs[ind, :],
          reduced_global_repeated_obs[ind, :],
          arm_single_obs[ind, :],
          reduced_arm_repeated_obs[ind, :],
      ))

    reward = np.concatenate(
        [self._reward_fn([concatenate_features(b) for b in batch_size_range])]
    )
    return reward