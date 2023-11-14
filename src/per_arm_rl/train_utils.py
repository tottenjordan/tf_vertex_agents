import pickle
import numpy as np
import tensorflow as tf
import logging
logging.disable(logging.WARNING)

from google.cloud import storage

from . import data_utils as data_utils
from . import data_config as data_config

project_number='hybrid-vertex'
storage_client = storage.Client(project=project_number)

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

# ====================================================
# get train & val datasets
# ====================================================
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
options.threading.max_intra_op_parallelism = 1 # TODO


def _get_train_dataset(
    bucket_name, 
    data_dir_prefix_path, 
    split, 
    total_take, 
    batch_size,
    num_replicas = 1,
    cache: bool = True,
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
    for blob in storage_client.list_blobs(f"{bucket_name}", prefix=f'{data_dir_prefix_path}/{split}'):
        if '.tfrecord' in blob.name:
            train_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))
            
    print(f"train_files: {train_files}")

    train_dataset = tf.data.TFRecordDataset(train_files)          # original
    # train_dataset = train_dataset.take(total_take)
    train_dataset = train_dataset.map(data_utils.parse_tfrecord)  # original
    
    # tmp - test start
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
    #     data_utils.parse_tfrecord, 
    #     num_parallel_calls=tf.data.AUTOTUNE
    # ).prefetch(
    #     tf.data.AUTOTUNE # GLOBAL_BATCH_SIZE*3 # tf.data.AUTOTUNE
    # ).with_options(
    #     options
    # )

    
    if cache:
        train_dataset = train_dataset.batch(batch_size).cache().repeat() # original
        # train_dataset.cache()
        # tmp - test end
    else:
        train_dataset = train_dataset.batch(batch_size).repeat()

    # train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    # train_dataset = train_dataset.cache()
    return train_dataset



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
    train_dataset = train_dataset.map(data_utils.parse_tfrecord) #, num_parallel_calls=tf.data.AUTOTUNE)
    
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
        logging.info('Restoring checkpoint from %s.', latest)
        checkpoint.restore(latest)
        logging.info('Successfully restored to step %s.', step_metric.result())
    else:
        logging.info(
            'Did not find a pre-existing checkpoint. Starting from scratch.'
        )
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
