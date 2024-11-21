"""A network that operates on repeated global and arm features."""
from typing import Callable, Sequence, Text

import gin
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.networks import nest_map
from tf_agents.networks import network
from tf_agents.networks import sequential
from tf_agents.specs import tensor_spec
from tf_agents.typing import types

GLOBAL_FEATURE_KEY = bandit_spec_utils.GLOBAL_FEATURE_KEY
PER_ARM_FEATURE_KEY = bandit_spec_utils.PER_ARM_FEATURE_KEY
SINGLE_FEATURE_KEY = 'single'
REPEATED_FEATURE_KEY = 'repeated'


def _check_reducer_correctness(
    reducer_fn: Callable[[types.Tensor], types.Tensor]
):
  shapes = ([3, 4, 5, 6], [9, 8, 7, 6, 5])
  for shape in shapes:
    input_tensor = tf.reshape(tf.range(np.prod(shape)), shape=shape)
    output_tensor = reducer_fn(input_tensor)
    new_shape = shape[:-2] + [shape[-1]]
    tf.debugging.assert_equal(
        output_tensor.shape,
        new_shape,
        message=(
            'Reducer produced unexpected output with input shape {}. '
            'The expected output shape is {}, but we got {}.'.format(
                shape, new_shape, output_tensor.shape.as_list()
            )
        ),
    )


def _broadcast_and_concatenate_fn(
    input_tensors: Sequence[types.Tensor],
) -> types.Tensor:
  """Takes 2 tensors, broadcasts the smaller one and concatenates them.

  This function is used in 3 layers:
    -- In both the global and per-arm branches it concatenates the single and
       repeated inputs.
    -- It concatenates the output of the global and the per-arm tower.

  Assumptions on the shapes: one of the inputs has rank 1 less than the other.
  The first r-1 dimensions of the smaller tensor has to match that of the
  second. The last dimensions of both tensors can be anything.

  An example of what this function does:
  input1.shape = [3, 4, 11]
  input2.shape = [3, 4, 5, 13]
  First, input1 gets a new dimension: [3, 4, 1, 11]. The it gets broadcast along
  the new dimension to get the shape [3, 4, 5, 11], where `5` is taken from the
  second to last dimension of input2. Finally, the two inputs get concatenated
  so that the resulting output has shape [3, 4, 5, 24].

  Args:
    input_tensors: a list of two tensors.

  Returns:
    A tensor.
  """
  ranks = [len(t.shape) for t in input_tensors]
  # We don't know the order of inputs after NestFlatten.
  if ranks[0] < ranks[1]:
    smaller = input_tensors[0]
    larger = input_tensors[1]
  else:
    smaller = input_tensors[1]
    larger = input_tensors[0]

  repetitions = tf.shape(larger)[-2]
  if repetitions is None:
    repetitions = 1
  old_smaller_shape = tf.shape(smaller)
  new_smaller_shape = tf.concat(
      [old_smaller_shape[:-1], [repetitions], old_smaller_shape[-1:]], axis=-1
  )
  broadcast_smaller = tf.broadcast_to(
      tf.expand_dims(smaller, axis=-2), new_smaller_shape
  )
  return tf.concat([broadcast_smaller, larger], axis=-1)


@gin.configurable
class RepeatedFeaturePerArmNetwork(network.Network):
  """A network for observations with repeated global and per-arm features.

  This class builds a network that takes structured observations as input, and
  outputs a reward estimate for every action and batch in the observation. To
  operate properly, it needs an observation spec as follows:
  obs_spec = {
    'global': {
      'single': TensorSpec(shape=[global_single_dim])
      'repeated': TensorSpec(shape=[global_repetitions, global_repeated_dim])
    },
    'per_arm': {
      'single': TensorSpec(shape=[num_actions, arm_single_dim])
      'repeated': TensorSpec(
          shape=[num_actions, arm_repetitions, arm_repeated_dim])
    }
  }
  The network that is built makes sure that the weights for the repeated
  features are shared accross repetitions.

  Usage: first initialize the network with
  ```
  net = RepeatedFeaturePerArmNetwork(obs_spec,
                                     global_layer_params,
                                     arm_layer_params,
                                     common_layer_params,
                                     reducer)
  ```

  The layer params should be lists of ints indicating the sizes of layers within
  the respective subnetwork. The reducer should be a function that takes a
  tensor as input and reduces it along the second to last dimension. That is, if
  the input is a tensor of shape `[a, b, c, d]`, it should return a tensor of
  shape `[a, b, d]`. The reduced dimension is the repetition dimension. Example:
  ```
  reducer = lambda t: tf.reduce_mean(t, axis=-2)
  ```

  Then create the network variables by calling `net.create_variables()`.
  The network can be called on a batched observation that adheres to the
  observation spec. It outputs a tensor of shape `[batch_size, num_actions]`.
  """

  def __init__(
      self,
      observation_spec: types.TensorSpec,
      global_layer_params: Sequence[int],
      arm_layer_params: Sequence[int],
      common_layer_params: Sequence[int],
      reducer: Callable[[types.Tensor], types.Tensor],
      name: Text = 'RepeatedFeaturePerArmNetwork',
  ):
    """Initializes a network with repeated features.

    Args:
      observation_spec: The observation spec that defines the syntax of the
        input.
      global_layer_params: The layer sizes of the global tower.
      arm_layer_params: The layer sizes of the arm tower.
      common_layer_params: The layer sizes of the common tower.
      reducer: A network that reduces the repetition dimensions.
      name: The name of this network instance.

    Returns:
      A network.
    """
    _check_reducer_correctness(reducer)
    self._repeated_dense_net = {}
    self._observation_spec = observation_spec
    global_tower = self._create_repeated_branch(
        global_layer_params, reducer, GLOBAL_FEATURE_KEY
    )
    arm_tower = self._create_repeated_branch(
        arm_layer_params, reducer, PER_ARM_FEATURE_KEY
    )
    self._common_dense_net = sequential.Sequential(
        [tf.keras.layers.Dense(i) for i in common_layer_params]
        + [
            tf.keras.layers.Dense(1, activation=None),
            tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=-1)),
        ],
        name='seq_common',
    )

    self._network = sequential.Sequential(
        [
            nest_map.NestMap({
                GLOBAL_FEATURE_KEY: global_tower,
                PER_ARM_FEATURE_KEY: arm_tower,
            }),
            nest_map.NestFlatten(),
            tf.keras.layers.Lambda(_broadcast_and_concatenate_fn),
            self._common_dense_net,
        ],
        name='seq_complete',
    )
    super(RepeatedFeaturePerArmNetwork, self).__init__(state_spec=(), name=name)

  def create_variables(self):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    """Creates the variables of the underlying dense towers.

    The `create_variables` member functions are called on each of the 3 towers
    with an input spec that does not contain the repeated dimensions, this way
    making sure that the weights are shared among repetitions.
    """
    branch_output_spec = {}
    for k, v in self._repeated_dense_net.items():
      branch_input_spec = self._observation_spec[k]
      input_dim = (
          branch_input_spec[SINGLE_FEATURE_KEY].shape[-1]
          + branch_input_spec[REPEATED_FEATURE_KEY].shape[-1]
      )
      dense_net_input_spec = tensor_spec.TensorSpec(
          shape=[input_dim], dtype=tf.float32
      )
      branch_output_spec[k] = v.create_variables(dense_net_input_spec)

    common_input_dim = sum(
        [spec.shape[-1] for spec in branch_output_spec.values()]
    )
    common_input_spec = tensor_spec.TensorSpec(
        shape=[common_input_dim], dtype=tf.float32
    )
    self._common_dense_net.create_variables(input_tensor_spec=common_input_spec)

  def call(self, observation, step_type=None, network_state=()):
    return self._network(
        observation, step_type=step_type, network_state=network_state
    )

  def _create_repeated_branch(
      self,
      layer_params: Sequence[int],
      reducer: Callable[[types.Tensor], types.Tensor],
      branch: Text,
  ) -> types.Network:
    """Creates either the global or the per-arm branch of the network.

    Args:
      layer_params: A sequence of ints containing layer sizes.
      reducer: A function that reduces the repetition dimension.
      branch: (string) Which branch is created. Valid values are `global` and
        `per_arm`.

    Returns:
      A network.
    """

    dense_network = sequential.Sequential(
        [tf.keras.layers.Dense(i) for i in layer_params], name='seq_' + branch
    )
    self._repeated_dense_net[branch] = dense_network
    combined_network = sequential.Sequential(
        [
            nest_map.NestFlatten(),
            tf.keras.layers.Lambda(_broadcast_and_concatenate_fn),
            dense_network,
            tf.keras.layers.Lambda(reducer),
        ],
        name='seq_wrap_' + branch,
    )

    return combined_network