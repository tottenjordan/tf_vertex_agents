import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pickle
import functools
import numpy as np
import tensorflow as tf

import logging
logging.disable(logging.WARNING)

from google.cloud import storage

# tf agents
from tf_agents.trajectories import trajectory
from tf_agents.bandits.policies import policy_utilities
from tf_agents.bandits.specs import utils as bandit_spec_utils

# this repo
from src.data import data_utils as data_utils
from src.data import data_config as data_config
from src.utils import train_utils as train_utils
from src.utils import reward_factory as reward_factory
# from src.networks import encoding_network as emb_features

storage_client = storage.Client(project=data_config.PROJECT_ID)

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
options.threading.max_intra_op_parallelism = 1

# ====================================================
# get TF Record Dataset function
# ====================================================
def example_proto_to_trajectory(
    example_proto,
):
    feature_description = {
        # target/label item features
        'target_movie_id': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        'target_movie_rating': tf.io.FixedLenFeature(shape=(), dtype=tf.float32),
        'target_rating_timestamp': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
        'target_movie_genres': tf.io.FixedLenFeature(shape=(data_config.MAX_GENRE_LENGTH), dtype=tf.string),
        'target_movie_year': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
        'target_movie_title': tf.io.FixedLenFeature(shape=(), dtype=tf.string),

        # user - global context features
        'user_id': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        'user_gender': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        'user_age': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
        'user_occupation_text': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        'user_zip_code': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
    }
    # single_example = tf.io.parse_single_example(example_proto, feature_description)
    parsed_example = tf.io.parse_single_sequence_example(example_proto, feature_description)
    
    return parsed_example[0]

def _trajectory_fn(element, hparams, embs):
    """Converts a dataset element into a trajectory."""
    global_features_emb = embs._get_global_context_features(element)
    arm_features_emb = embs._get_per_arm_features(element)
    
    # Adds a time dimension.
    arm_features = train_utils._add_outer_dimension(arm_features_emb)
    observation = {
        bandit_spec_utils.GLOBAL_FEATURE_KEY:
            # train_utils._add_outer_dimension(tf.concat(global_features_emb, axis=1))
            train_utils._add_outer_dimension(global_features_emb),
    }
    reward = train_utils._add_outer_dimension(reward_factory._get_rewards(element))
    # reward = train_utils._add_outer_dimension(reward_factory._get_binary_rewards(element))
    
    dummy_rewards = tf.zeros([hparams['batch_size'], 1, hparams['num_actions']])
    policy_info = policy_utilities.PerArmPolicyInfo(
        chosen_arm_features=arm_features,
        predicted_rewards_mean=dummy_rewards,
        # bandit_policy_type=tf.zeros([hparams['batch_size'], 1, 1], dtype=tf.int32)
    )
    if hparams['model_type'] == 'NeuralLinUCB':
        policy_info = policy_info._replace(
            predicted_rewards_optimistic=dummy_rewards
        )
    
    return trajectory.single_step(
        observation=observation,
        action=tf.zeros_like(
            reward, dtype=tf.int32
        ),  # Arm features are copied from policy info, put dummy zeros here
        policy_info=policy_info,
        reward=reward,
        discount=tf.zeros_like(reward))

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
    # ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def create_tfrecord_ds(
    filenames,
    process_example_fn,
    process_trajectory_fn,
    batch_size: int,
    shuffle_buffer_size_per_record: int = 1,
    shuffle_buffer_size: int = 1024,
    num_shards: int = 50,
    cycle_length: int = tf.data.AUTOTUNE,
    block_length: int = tf.data.AUTOTUNE,
    num_prefetch: int = 10,
    num_parallel_calls: int = 10,
    repeat: bool = True,
    drop_remainder: bool = False,
    # cache: bool = True
):
    """
      Each element of the TFRecord data is parsed using the process_example_fn
      and converted to Tensors. A dataset is created for each record file and these
      are interleaved together to create the final dataset.
    """
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

    # filename_ds = filename_ds.shuffle(len(filenames))    
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
    # parsed dataset; not trajectory yet
    example_ds = example_ds.shuffle(shuffle_buffer_size)
    
    # map trajectory
    example_ds = example_ds.batch(
        batch_size,
        drop_remainder=drop_remainder
    ).map(
        process_trajectory_fn,
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    
    # TODO: test cache impact to perf
    # if cache:
    #     example_ds.cache()
  
    return example_ds


# ====================================================
# get train & val datasets
# TODO: replace with "create TF Record dataset function
# ====================================================
def _get_train_dataset(
    bucket_name, 
    data_dir_prefix_path, 
    split,  
    batch_size,
    num_replicas = 1,
    cache: bool = True,
    is_testing: bool = False,
):
    """
    TODO: use `dataset.take(k).cache().repeat()`
    """
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    # Disable intra-op parallelism to optimize for throughput instead of latency.
    options.threading.max_intra_op_parallelism = 1 # TODO 
    
    GLOBAL_BATCH_SIZE = int(batch_size) * int(num_replicas)
    logging.info(f'GLOBAL_BATCH_SIZE = {GLOBAL_BATCH_SIZE}')
    
    train_files = []
    for blob in storage_client.list_blobs(
        f"{bucket_name}", 
        prefix=f'{data_dir_prefix_path}/{split}'
    ):
        if '.tfrecord' in blob.name:
            train_files.append(
                blob.public_url.replace(
                    "https://storage.googleapis.com/", "gs://"
                )
            )
    if is_testing:
        train_files = train_files[:3]
    print(f"number of train_files: {len(train_files)}")

    train_dataset = tf.data.TFRecordDataset(train_files)
    train_dataset = train_dataset.map(data_utils._parse_function)
    if cache:
        train_dataset = train_dataset.batch(batch_size).cache().repeat()
    else:
        train_dataset = train_dataset.batch(batch_size).repeat()

    return train_dataset
    
    # ### tmp - test start
    # train_dataset = tf.data.Dataset.from_tensor_slices(train_files).prefetch(
    #     tf.data.AUTOTUNE,
    # )
    # train_dataset = train_dataset.interleave( # Parallelize data reading
    #     data_utils.full_parse,
    #     cycle_length=tf.data.AUTOTUNE,
    #     block_length=64,
    #     num_parallel_calls=tf.data.AUTOTUNE,
    #     deterministic=False
    # ).repeat().batch( #vectorize mapped function
    #     GLOBAL_BATCH_SIZE,
    #     drop_remainder=True,
    # ).map(
    #     data_utils._parse_function, 
    #     num_parallel_calls=tf.data.AUTOTUNE
    # ).prefetch(
    #     tf.data.AUTOTUNE # GLOBAL_BATCH_SIZE*3 # tf.data.AUTOTUNE
    # ).with_options(
    #     options
    # )
    ### tmp - test end

    # train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    # train_dataset = train_dataset.cache()

    # return train_dataset


def _get_eval_dataset(bucket_name, data_dir_prefix_path, split, batch_size):
    train_files = []
    for blob in storage_client.list_blobs(
        f"{bucket_name}", prefix=f'{data_dir_prefix_path}/{split}'
    ):
        if '.tfrecord' in blob.name:
            train_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))
            
    logging.info(f"train_files: {train_files}")

    train_dataset = tf.data.TFRecordDataset(train_files)
    # train_dataset.repeat().batch(batch_size)
    
    # train_dataset = train_dataset.map(movielens_ds_utils.parse_tfrecord)
    train_dataset = train_dataset.map(data_utils._parse_function)
    #, num_parallel_calls=tf.data.AUTOTUNE)
    
    return train_dataset


# ====================================================
# misc. helper functions
# ====================================================
# os.mkdir('trajectories')

def save_trajectories(trajectory, filename):
    # Open a file for writing
    with open(f"{filename}.p", "wb") as f:
        # Write the dictionary to the file
        pickle.dump(trajectory, f)

    # Close the file
    f.close()
    
def get_arch_from_string(arch_string):
    
    q = arch_string.replace(']', '')
    q = q.replace('[', '')
    q = q.replace(" ", "")
    
    return [
        int(x) for x in q.split(',')
    ]

# ====================================================
# distributed training with Vertex AI
# ====================================================

def _is_chief(task_type, task_id): 
    ''' Check for primary if multiworker training
    '''
    if task_type == 'chief':
        results = 'chief'
    else:
        results = None
    return results

def get_train_strategy(distribute_arg):

    # Single Machine, single compute device
    if distribute_arg == 'single':
        if tf.config.list_physical_devices('GPU'):
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        else:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        logging.info("Single device training")  
    # Single Machine, multiple compute device
    elif distribute_arg == 'mirrored':
        strategy = tf.distribute.MirroredStrategy()
        logging.info("Mirrored Strategy distributed training")
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    # Multi Machine, multiple compute device
    elif distribute_arg == 'multiworker':
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        logging.info("Multi-worker Strategy distributed training")
        logging.info('TF_CONFIG = {}'.format(os.environ.get('TF_CONFIG', 'Not found')))
    # Single Machine, multiple TPU devices
    elif distribute_arg == 'tpu':
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)
        logging.info("All devices: ", tf.config.list_logical_devices('TPU'))

    return strategy

def prepare_worker_pool_specs(
    image_uri,
    args,
    replica_count=1,
    machine_type="n1-standard-16",
    accelerator_count=1,
    accelerator_type="ACCELERATOR_TYPE_UNSPECIFIED",
    reduction_server_count=0,
    reduction_server_machine_type = "n1-highcpu-16",
    reduction_server_image_uri = "us-docker.pkg.dev/vertex-ai-restricted/training/reductionserver:latest",
):

    if accelerator_count > 0:
        machine_spec = {
            "machine_type": machine_type,
            "accelerator_type": accelerator_type,
            "accelerator_count": accelerator_count,
        }
    else:
        machine_spec = {"machine_type": machine_type}

    container_spec = {
        "image_uri": image_uri,
        "args": args,
    }

    chief_spec = {
        "replica_count": 1,
        "machine_spec": machine_spec,
        "container_spec": container_spec,
    }

    worker_pool_specs = [chief_spec]
    if replica_count > 1:
        workers_spec = {
            "replica_count": replica_count - 1,
            "machine_spec": machine_spec,
            "container_spec": container_spec,
        }
        worker_pool_specs.append(workers_spec)
    if reduction_server_count > 1:
        workers_spec = {
            "replica_count": reduction_server_count,
            "machine_spec": {
                "machine_type": reduction_server_machine_type,
            },
            "container_spec": {"image_uri": reduction_server_image_uri},
        }
        worker_pool_specs.append(workers_spec)

    return worker_pool_specs

# ====================================================
# metric summaries
# ====================================================

from tf_agents.eval import metric_utils
from tf_agents.metrics import export_utils

def _export_metrics_and_summaries(step, metrics):
    """Exports metrics and tf summaries."""
    metric_utils.log_metrics(metrics)
    export_utils.export_metrics(step=step, metrics=metrics)
    for metric in metrics:
        metric.tf_summaries(train_step=step)
        
# ====================================================
# helper functions: shapes and dims
# ====================================================
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.networks import encoding_network
from tf_agents.networks import network
from tf_agents.networks import q_network
from tf_agents.specs import tensor_spec
from tf_agents.typing import types

def _remove_num_actions_dim_from_spec(
    observation_spec: types.NestedTensorSpec
) -> types.NestedTensorSpec:
    """
    Removes the extra `num_actions` 
    dimension from the observation spec.
    """
    obs_spec_no_num_actions = {
        bandit_spec_utils.GLOBAL_FEATURE_KEY:
            observation_spec[bandit_spec_utils.GLOBAL_FEATURE_KEY],
        
        bandit_spec_utils.PER_ARM_FEATURE_KEY:
            tensor_spec.remove_outer_dims_nest(
                observation_spec[bandit_spec_utils.PER_ARM_FEATURE_KEY], 1
            )
    }
    
    if bandit_spec_utils.NUM_ACTIONS_FEATURE_KEY in observation_spec:
        obs_spec_no_num_actions.update(
            {
                bandit_spec_utils.NUM_ACTIONS_FEATURE_KEY:
                    observation_spec[bandit_spec_utils.NUM_ACTIONS_FEATURE_KEY]
            }
        )
    return obs_spec_no_num_actions


def _add_outer_dimension(x):
    """Adds an extra outer dimension."""
    if isinstance(x, dict):
        for key, value in x.items():
            x[key] = tf.expand_dims(value, 1)
        return x
    return tf.expand_dims(x, 1)


def _remove_outer_dimension(x):
    return tf.nest.map_structure(lambda t: tf.squeeze(t, 1), x)


def _as_multi_dim(maybe_scalar):
    if maybe_scalar is None:
        shape = ()
    elif tf.is_tensor(maybe_scalar) and maybe_scalar.shape.rank > 0:
        shape = maybe_scalar
    elif np.asarray(maybe_scalar).ndim > 0:
        shape = maybe_scalar
    else:
        shape = (maybe_scalar,)
    return shape


def _as_array(a, t=np.float32):
    if t is None:
        t = np.float32
    r = np.asarray(a, dtype=t)
    if np.isnan(np.sum(r)):
        raise ValueError(
            'Received a time_step input that converted to a nan array.'
            ' Did you accidentally set some input value to None?.\n'
            'Got:\n{}'.format(a)
        )
    return r

# ====================================================
# train loop functions
# ====================================================
from tf_agents.bandits.replay_buffers import bandit_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.eval import metric_utils
from tf_agents.metrics import export_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver
from tf_agents.trajectories import time_step as ts

def _get_replay_buffer(
    data_spec, 
    batch_size, 
    steps_per_loop,
    async_steps_per_loop
):
    """
    Return a `TFUniformReplayBuffer` for the given `agent`.
    """
    return bandit_replay_buffer.BanditReplayBuffer(
        data_spec=data_spec,
        batch_size=batch_size,
        max_length=steps_per_loop * async_steps_per_loop
    )

def set_expected_shape(experience, num_steps):
    """Sets expected shape."""

    def set_time_dim(input_tensor, steps):
        tensor_shape = input_tensor.shape.as_list()
        if len(tensor_shape) < 2:
            raise ValueError(
                'input_tensor is expected to be of rank-2, but found otherwise: '
                f'input_tensor={input_tensor}, tensor_shape={tensor_shape}'
            )
        tensor_shape[1] = steps
        input_tensor.set_shape(tensor_shape)

    tf.nest.map_structure(lambda t: set_time_dim(t, num_steps), experience)

# ====================================================
# checkpoint manager
# ====================================================
AGENT_CHECKPOINT_NAME = 'agent'
STEP_CHECKPOINT_NAME = 'step'
CHECKPOINT_FILE_PREFIX = 'ckpt'

def restore_and_get_checkpoint_manager(
    root_dir, 
    agent, 
    metrics, 
    step_metric
):
    """
    Restores from `root_dir` and returns a function that writes checkpoints.
    """
    trackable_objects = {metric.name: metric for metric in metrics}
    trackable_objects[AGENT_CHECKPOINT_NAME] = agent
    trackable_objects[STEP_CHECKPOINT_NAME] = step_metric
    checkpoint = tf.train.Checkpoint(**trackable_objects)
    checkpoint_manager = tf.train.CheckpointManager(
      checkpoint=checkpoint, directory=root_dir, max_to_keep=5
    )
    latest = checkpoint_manager.latest_checkpoint
    if latest is not None:
        print(f'Restoring checkpoint from {latest}')
        checkpoint.restore(latest)
        print(f'Successfully restored to step {step_metric.numpy()}')
    else:
        print('Did not find a pre-existing checkpoint. Starting from scratch.')
    return checkpoint_manager

# ====================================================
# get inference trajectory
# ====================================================
def _get_infer_step(
    feature, 
    batch_size, 
    rewards
):
    shape = _as_multi_dim(batch_size)
    
    step_type_tf = tf.fill(
        shape, ts.StepType.FIRST, name='step_type'
    )
    
    discount_tf = tf.fill(
        shape, _as_array(1.0), name='discount'
    )
    
    infer_step = ts.TimeStep(
        step_type_tf,
        rewards,
        discount_tf,
        feature
    )
    
    return infer_step
    
# ====================================================
# get eval trajectory
# ====================================================
def _get_eval_step(
    feature, 
    reward_np
):
    
    infer_step = ts.TimeStep(
        tf.constant(
            ts.StepType.FIRST, 
            dtype=tf.int32, 
            shape=[],
            name='step_type'
        ),
        tf.constant(
            reward_np, dtype=tf.float32, shape=[], name='reward'
        ),
        tf.constant(
            1.0, dtype=tf.float32, shape=[], name='discount'
        ),
        feature
    )
    
    return infer_step

# ====================================================
# converting tensor specs
# ====================================================
from tensorflow.python.framework import tensor_spec as tspecs
from tf_agents.specs import array_spec

TensorSpec = tspecs.TensorSpec
BoundedTensorSpec = tspecs.BoundedTensorSpec

def is_bounded(spec):
    if isinstance(spec, (array_spec.BoundedArraySpec, BoundedTensorSpec)):
        return True
    elif hasattr(spec, "minimum") and hasattr(spec, "maximum"):
        return hasattr(spec, "dtype") and hasattr(spec, "shape")

def from_spec(spec):
    """
    Maps the given spec into corresponding TensorSpecs keeping bounds.
    """

    def _convert_to_tensor_spec(s):
        # Need to check bounded first as non bounded specs are base class.
        if isinstance(s, tf.TypeSpec):
            return s
        if is_bounded(s):
            return BoundedTensorSpec.from_spec(s)
        elif isinstance(s, array_spec.ArraySpec):
            return TensorSpec.from_spec(s)
        else:
            raise ValueError(
                "No known conversion from type `%s` to a TensorSpec.  Saw:\n  %s" % (type(s), s)
            )

    return tf.nest.map_structure(_convert_to_tensor_spec, spec)

# ====================================================
# environemnt utils
# ====================================================
def compute_optimal_reward_with_my_environment(observation, environment):
    """Helper function for gin configurable Regret metric."""
    del observation
    return tf.py_function(environment.compute_optimal_reward, [], tf.float32)

def compute_optimal_action_with_my_environment(
    observation, 
    environment, 
    action_dtype=tf.int32
):
    """Helper function for gin configurable SuboptimalArms metric."""
    del observation
    return tf.py_function(environment.compute_optimal_action, [], action_dtype)