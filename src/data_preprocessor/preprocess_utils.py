import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tf_agents import trajectories
from tf_agents.trajectories import trajectory
from tf_agents.bandits.policies import policy_utilities
from tf_agents.bandits.specs import utils as bandit_spec_utils

from typing import Callable, Dict, List, Optional, TypeVar, Any

from google.cloud import bigquery

def build_dict_from_trajectory(
    trajectory : trajectories.Trajectory
) -> Dict[str, Any]:
    trajectory_dict = {
        "step_type": trajectory.step_type.numpy().tolist(),
        "observation": [
            {
                "observation_batch": batch
            } for batch in trajectory.observation['global'].numpy().tolist()
        ],
        "chosen_arm_features": [
            {
                "chosen_arm_features_batch": batch
            } for batch in trajectory.policy_info.chosen_arm_features.numpy().tolist()
        ],
        "action": trajectory.action.numpy().tolist(),
        "next_step_type": trajectory.next_step_type.numpy().tolist(),
        "reward": trajectory.reward.numpy().tolist(),
        "discount": trajectory.discount.numpy().tolist(),
    }
    return trajectory_dict

job_config = bigquery.LoadJobConfig(
    schema=[
        bigquery.SchemaField("step_type", "INT64", mode="REPEATED"),
        bigquery.SchemaField(
            "observation",
            "RECORD",
            mode="REPEATED",
            fields=[
                bigquery.SchemaField("observation_batch", "FLOAT64",
                                     "REPEATED")
            ]),
        bigquery.SchemaField(
            "chosen_arm_features",
            "RECORD",
            mode="REPEATED",
            fields=[
                bigquery.SchemaField("chosen_arm_features_batch", "FLOAT64",
                                     "REPEATED")
            ]),
        bigquery.SchemaField("action", "INT64", mode="REPEATED"),
        bigquery.SchemaField("next_step_type", "INT64", mode="REPEATED"),
        bigquery.SchemaField("reward", "FLOAT64", mode="REPEATED"),
        bigquery.SchemaField("discount", "FLOAT64", mode="REPEATED"),
    ],
    source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
)

def _bytes_feature(tensor: tf.Tensor) -> tf.train.Feature:
    """
    Returns a `tf.train.Feature` with bytes from `tensor`.

    Args:
      tensor: A `tf.Tensor` object.

    Returns:
      A `tf.train.Feature` object containing bytes that represent the content of
      `tensor`.
    """
    value = tf.io.serialize_tensor(tensor)
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value])
    )

def build_example_from_bq(
    data_row: bigquery.table.Row
) -> tf.train.Example:
    """
    Builds a `tf.train.Example` from `data_row` content.

    Args:
      data_row: 
        A `bigquery.table.Row` object that contains 7 pieces of data:
          `step_type`, `observation`, `chosen_arm_features`, `action`, `policy_info`,
          `next_step_type`,`reward`, `discount`. Each piece of data except `observation`
          and `chosen_arm_features` are 1D arrays; 
          `observation` is a 1D array of `{"observation_batch": 1D array}.`
          `chosen_arm_features` is a 1D array of `{"chosen_arm_features_batch": 1D array}.`

    Returns:
      A `tf.train.Example` object holding the same data as `data_row`.
    """
    feature = {
        "observation":
            _bytes_feature([
                observation["observation_batch"]
                for observation in data_row.get("observation")
            ]),
        "chosen_arm_features":
            _bytes_feature([
                arm_feats["chosen_arm_features_batch"]
                for arm_feats in data_row.get("chosen_arm_features")
            ]),
        "step_type":
            _bytes_feature(data_row.get("step_type")),
        "action":
            _bytes_feature(data_row.get("action")),
        "next_step_type":
            _bytes_feature(data_row.get("next_step_type")),
        "reward":
            _bytes_feature(data_row.get("reward")),
        "discount":
            _bytes_feature(data_row.get("discount")),
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto

def write_tfrecords(
    tfrecord_file: str,
    table: bigquery.table.RowIterator
) -> None:
    """
    Writes the row data in `table` into TFRecords in `tfrecord_file`.

    Args:
      tfrecord_file: Path to file to write the TFRecords.
      table: A row iterator over all data to be written.
    """
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for data_row in table:
            example = build_example_from_bq(data_row)
            writer.write(
                example.SerializeToString()
            )
            
# Mapping from feature name to serialized value
feature_description = {
    "observation": tf.io.FixedLenFeature((), tf.string),
    "chosen_arm_features": tf.io.FixedLenFeature((), tf.string),
    "step_type": tf.io.FixedLenFeature((), tf.string),
    "action": tf.io.FixedLenFeature((), tf.string),
    "next_step_type": tf.io.FixedLenFeature((), tf.string),
    "reward": tf.io.FixedLenFeature((), tf.string),
    "discount": tf.io.FixedLenFeature((), tf.string),
}

def _parse_record(raw_record: tf.Tensor) -> Dict[str, tf.Tensor]:
    """
    Parses a serialized `tf.train.Example` proto.

    Args:
      raw_record: A serialized data record of a `tf.train.Example` proto.

    Returns:
      A dict mapping feature names to values as `tf.Tensor` objects of type
      string containing serialized protos, following `feature_description`.
    """
    return tf.io.parse_single_example(raw_record, feature_description)

def build_trajectory_from_tfrecord(
    parsed_record: Dict[str, tf.Tensor],
    batch_size: int,
    num_actions: int,
    # policy_info: policies.utils.PolicyInfo
) -> trajectories.Trajectory:
    """
    Builds a `trajectories.Trajectory` object from `parsed_record`.

    Args:
      parsed_record: A dict mapping feature names to values as `tf.Tensor`
        objects of type string containing serialized protos.
      policy_info: Policy information specification.

    Returns:
      A `trajectories.Trajectory` object that contains values as de-serialized
      `tf.Tensor` objects from `parsed_record`.
    """
    dummy_rewards = tf.zeros([batch_size, 1, num_actions])

    global_features = tf.expand_dims(
        tf.io.parse_tensor(parsed_record["observation"], out_type=tf.float32),
        axis=1
    )
    observation = {
        bandit_spec_utils.GLOBAL_FEATURE_KEY: global_features
    }
    
    arm_features = tf.expand_dims(
        tf.io.parse_tensor(parsed_record["chosen_arm_features"], out_type=tf.float32),
        axis=1
    )

    policy_info = policy_utilities.PerArmPolicyInfo(
        chosen_arm_features=arm_features,
        predicted_rewards_mean=dummy_rewards,
        bandit_policy_type=tf.zeros([batch_size, 1, 1], dtype=tf.int32)
    )

    return trajectories.Trajectory(
        step_type=tf.expand_dims(
            tf.io.parse_tensor(parsed_record["step_type"], out_type=tf.int32),
            axis=1
        ),
        observation = observation,
        action=tf.expand_dims(
            tf.io.parse_tensor(parsed_record["action"], out_type=tf.int32),
            axis=1
        ),
        policy_info=policy_info,
        next_step_type=tf.expand_dims(
            tf.io.parse_tensor(
                parsed_record["next_step_type"], out_type=tf.int32),
            axis=1
        ),
        reward=tf.expand_dims(
            tf.io.parse_tensor(parsed_record["reward"], out_type=tf.float32),
            axis=1
        ),
        discount=tf.expand_dims(
            tf.io.parse_tensor(parsed_record["discount"], out_type=tf.float32),
            axis=1
        )
    )