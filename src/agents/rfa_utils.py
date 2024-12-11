"""
utils for the REINFORCE Recommender Agent
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import functools
import numpy as np
import tensorflow as tf
from typing import Sequence

from tensorflow.python.ops import nn_impl

from tf_agents.typing import types
from tf_agents.networks import sequential
from tf_agents.networks import utils as network_utils
from tf_agents.keras_layers import dynamic_unroll_layer

from tf_agents.trajectories import trajectory
from tf_agents.trajectories import time_step as ts

from src.data import data_config as data_config

IntegerLookup = tf.keras.layers.IntegerLookup
StringLookup = tf.keras.layers.StringLookup
KERAS_LSTM_FUSED = 2

def softmax_log_prob(
    weights: types.Float,
    biases: types.Float,
    classes: types.Int,
    inputs: types.Float,
) -> types.Float:
    """
    Computes softmax log probabilties.

    Args:
      weights: A [num_classes, embedding_size] tensor of class embeddings.
      biases: A Tensor of shape [num_classes]. The class biases.
      classes: A Tensor of type int64 and shape [B] or [B, T] representing the
        chosen classes.
      inputs: A Tensor of shape [B, embedding_size] or [B, T, embedding_size]. The
        output of the input embedding network.

    Returns:
      log_probs: A float tensor of the log probabilties for the chosen
        `classes`, having the same shape as `classes`.
    """
    logits = tf.matmul(inputs, weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, biases)
    log_probs = -tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=classes, logits=logits
    )
    return log_probs


def sampled_softmax_log_prob(
    weights: types.Float,
    biases: types.Float,
    classes: types.Int,
    inputs: types.Float,
    num_sampled: int,
    num_classes: int,
    outer_dims: int,
) -> types.Float:
    """
    Computes sampled output training log probabilties as in sampled softmax.

    Args:
      weights: A [num_classes, embedding_size] tensor of class embeddings.
      biases: A Tensor of shape [num_classes]. The class biases.
      classes: A Tensor of type int64 and shape [B] or [B, T] representing the
        chosen classes.
      inputs: A Tensor of shape [B, embedding_size] or [B, T, embedding_size]. The
        output of the input embedding network.
      num_sampled: The number of classes to randomly sample per batch.
      num_classes: The number of possible classes.
      outer_dims: Number of outer dims in `classes` or `inputs`. This is 2 if `[B,
        T]` or 1 if `[B]`.

    Returns:
      log_probs: A float tensor of the log probabilties for the chosen
        `classes`, having the same shape as `classes`.
    """
    batch_squash = network_utils.BatchSquash(outer_dims)
    classes = batch_squash.flatten(classes)
    inputs = batch_squash.flatten(inputs)

    # TODO: uniform sampler (JT)
    logits, _ = nn_impl._compute_sampled_logits(
        weights,
        biases,
        tf.expand_dims(classes, -1),
        inputs,
        num_sampled,
        num_classes,
    )
    
    normalized_logits = normalize_sampled_logits(logits)
    return batch_squash.unflatten(normalized_logits)

def normalize_sampled_logits(
    logits: types.Float,
) -> types.Float:
    """
    Computes normalized logits (log probabilties) from sampled softmax.

    Args:
      logits: `[B, 1 + num_sampled]` float tensor of unnormalized logits, where
        the first column contains the logits of the chosen or 'true' class, and
        the rest are the sampled or 'false' logits.

    Returns:
      normalized_logits: A `[B]` tensor of normalized logits for the chosen/true
        classes
    """
    # TODO: JT
    max_logits = tf.reduce_max(logits, axis=-1, keepdims=True)
    logits -= max_logits
    true_logits = logits[:, 0]
    log_sum_exp_logits = tf.math.log(tf.reduce_sum(tf.math.exp(logits), axis=-1))
    
    return true_logits - log_sum_exp_logits

# ==============================================
# state embedding network
# ==============================================
dense = functools.partial(
    tf.keras.layers.Dense,
    activation=tf.keras.activations.relu,
    kernel_initializer=tf.compat.v1.variance_scaling_initializer(
        scale=2.0, 
        mode='fan_in', 
        distribution='truncated_normal'
    )
)
fused_lstm_cell = functools.partial(
    tf.keras.layers.LSTMCell, 
    implementation=KERAS_LSTM_FUSED
)
embedding = functools.partial(tf.keras.layers.Embedding, input_length=1)

def create_state_embedding_network(
    observation_lookup_layer: IntegerLookup, # StringLookup | IntegerLookup
    input_embedding_size: int,
    input_fc_layer_units: Sequence[int],
    lstm_size: Sequence[int],
    output_fc_layer_units: Sequence[int]):
    """
    Creates an network to compute state embeddings from observations.
    """
    vocabulary_size = observation_lookup_layer.vocab_size()
    rnn_cell = tf.keras.layers.StackedRNNCells(
        [fused_lstm_cell(s) for s in lstm_size]
    )
    
    return sequential.Sequential(
        [observation_lookup_layer] 
        + [embedding(vocabulary_size, input_embedding_size)]
        + [dense(num_units) for num_units in input_fc_layer_units]
        + [dynamic_unroll_layer.DynamicUnroll(rnn_cell)]
        + [dense(num_units) for num_units in output_fc_layer_units]
    )

# ==============================================
# data utils 
# ==============================================
def create_tfrecord_ds(
    filenames,
    process_example_fn,
    batch_size: int,
    shuffle_buffer_size_per_record: int = 1,
    shuffle_buffer_size: int = 10000,
    num_shards: int = 50,
    cycle_length: int = tf.data.AUTOTUNE,
    block_length: int = 10,
    num_prefetch: int = 10,
    num_parallel_calls: int = 10,
    repeat: bool = True,
    drop_remainder: bool = False
):
    filenames = list(filenames)
    initial_len = len(filenames)
    remainder = initial_len % num_shards
    
    for _ in range(num_shards - remainder):
        filenames.append(
            filenames[np.random.randint(low=0, high=initial_len)]
        )
        
    filenames = np.array(filenames)
    np.random.shuffle(filenames)
    filenames = np.array_split(filenames, num_shards)
    filename_ds = tf.data.Dataset.from_tensor_slices(filenames)
    
    if repeat:
        filename_ds = filename_ds.repeat()
    
    filename_ds = filename_ds.shuffle(len(filenames))
    
    example_ds = filename_ds.interleave(
        functools.partial(
            create_single_tfrecord_ds,
            process_example_fn=process_example_fn,
            shuffle_buffer_size=shuffle_buffer_size_per_record,
        ),
        cycle_length=tf.data.AUTOTUNE,
        block_length=block_length,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    example_ds = example_ds.shuffle(shuffle_buffer_size)
    
    example_ds = example_ds.batch(
        batch_size, drop_remainder=drop_remainder
    ).prefetch(num_prefetch)
  
    return example_ds

def create_single_tfrecord_ds(
    filename,
    process_example_fn,
    shuffle_buffer_size = 1,
):
    raw_ds = tf.data.TFRecordDataset(filename)
    
    ds = raw_ds.map(
        process_example_fn,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def example_proto_to_trajectory(
    example_proto, # sequence_feature,
    sequence_length: int
):
    """
    Converts a sequence example to a Trajectory and weights for training.

    For now, we are using the following simplified features. At every point in
    time, the `context_movie_id` field in the sequence is the action and the `context_movie_id`
    at the previous time step (last action) is the observation. The `context_movie_rating` field
    is converted to a binary reward.

    If the sequence example is longer than than `sequence_length`, we only take
    the last part of the sequence example. If it is shorter, we pad it with dummy
    values at the end to equal `sequence_length`.

    Args:
    sequence_feature: A serialized SequenceExample to convert to a
      trajectory.
    sequence_length: The time dimension of the returned trajectory.

    Returns:
    trajectory: An unbatched trajectory. The time dimension will be equal to
      sequence length. The agent assumes that this trajectory is a single
      episode, so `trajectory.step_type` and `trajectory.discount` are ignored.
    weights: A [T] float tensor of weights. Each row of `weights`
        (along the time dimension) is usually a sequence of 0's, followed by
        a sequence of 1's, again followed by a sequence of 0's. This divides
        the trajectory into 3 parts. The first part is used to warm start
        the state embedding network. The second part is used to compute
        losses. Returns are computed using the second and third parts.
    """
    
    feature_description = {
        'context_movie_id': tf.io.FixedLenFeature(shape=(data_config.MAX_CONTEXT_LENGTH), dtype=tf.string),
        'context_movie_rating': tf.io.FixedLenFeature(shape=(data_config.MAX_CONTEXT_LENGTH), dtype=tf.float32),
    }
    
    sequence_feature = tf.io.parse_single_sequence_example(example_proto, feature_description)
    
    context_id_int = tf.strings.to_number(
        sequence_feature[0]['context_movie_id'],
        out_type=tf.dtypes.int64,
        name=None
    )
    
    sequence_feature[0]['context_movie_id'] = context_id_int
    actions = sequence_feature[0]['context_movie_id'][-sequence_length:]
    rewards = sequence_feature[0]['context_movie_rating'][-sequence_length:]
    observations = sequence_feature[0]['context_movie_id'][-(sequence_length+1):-1]

    # actual length
    actual_sequence_length = tf.shape(observations)[0]
    
    actions = actions[-actual_sequence_length:]
    rewards = rewards[-actual_sequence_length:]

    # padding
    paddings = tf.stack([0, sequence_length - actual_sequence_length])
    paddings = tf.expand_dims(paddings, 0)

    rewards = tf.pad(rewards, paddings, 'CONSTANT', constant_values=0)
    actions = tf.pad(actions, paddings, 'CONSTANT', constant_values=0)
    observations = tf.pad(observations, paddings, 'CONSTANT', constant_values=0)

    # steps & discounts
    discounts = tf.ones((sequence_length,), dtype=tf.float32)
    next_step_types = tf.ones(
      (sequence_length,), dtype=tf.int32) * ts.StepType.MID
    step_types = tf.concat([[ts.StepType.FIRST], next_step_types[1:]], axis=0)

    # build trajectory
    traj = trajectory.Trajectory(
        step_type=step_types,
        observation=observations,
        action=actions,
        policy_info=(),
        next_step_type=next_step_types,
        reward=rewards,
        discount=discounts
    )

    # get importance weights
    section_size = tf.cast(actual_sequence_length / 3, tf.int32)
    # print(f"section_size: {section_size}")
    
    weights = tf.concat(
        [
            tf.zeros((section_size,)),
            tf.ones((section_size,)),
            tf.zeros((sequence_length - 2 * section_size,))
        ], 
        axis=0
    )
    
    return traj, weights # sequence_feature