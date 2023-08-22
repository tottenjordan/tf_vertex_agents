# ====================================================
# train loop 1
# ====================================================
import os
import time
import logging
import pickle as pkl
from pprint import pprint
from datetime import datetime
from collections import defaultdict
from typing import Callable, Dict, List, Optional, TypeVar

import collections
from tf_agents.utils import common
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.trajectories import trajectory

import tensorflow as tf

# TF-Agents
from tf_agents.metrics.tf_metric import TFStepMetric
from tf_agents.eval import metric_utils
from tf_agents.metrics import export_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.train.utils import strategy_utils
from tf_agents.drivers import dynamic_step_driver

from tf_agents.policies import py_tf_eager_policy
from tf_agents.train.utils import strategy_utils

# logging
import logging
logging.disable(logging.WARNING)

from google.cloud import aiplatform as vertex_ai
from google.cloud import storage

# this repo
from src.per_arm_rl import train_utils
from src.per_arm_rl import data_utils
from src.per_arm_rl import data_config

if tf.__version__[0] != "2":
    raise Exception("The trainer only runs with TensorFlow version 2.")


PER_ARM = True  # Use the non-per-arm version of the MovieLens environment.

# clients
# project_number = os.environ["CLOUD_ML_PROJECT_ID"]
project_number='hybrid-vertex'
storage_client = storage.Client(project=project_number)

# ====================================================
# metrics
# ====================================================
def _export_metrics_and_summaries(step, metrics):
    """Exports metrics and tf summaries."""
    metric_utils.log_metrics(metrics)
    export_utils.export_metrics(step=step, metrics=metrics)
    for metric in metrics:
        metric.tf_summaries(train_step=step)
        
# ====================================================
# checkpoint manager
# ====================================================
AGENT_CHECKPOINT_NAME = 'agent'
STEP_CHECKPOINT_NAME = 'step'
CHECKPOINT_FILE_PREFIX = 'ckpt'

def restore_and_get_checkpoint_manager(root_dir, agent, metrics, step_metric):
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
        print('Restoring checkpoint from %s.', latest)
        checkpoint.restore(latest)
        print('Successfully restored to step %s.', step_metric.result())
    else:
        print(
            'Did not find a pre-existing checkpoint. Starting from scratch.'
        )
    return checkpoint_manager

# ====================================================
# get train & val datasets
# ====================================================
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO

def _get_train_dataset(bucket_name, data_dir_prefix_path, split, batch_size):
    train_files = []
    for blob in storage_client.list_blobs(f"{bucket_name}", prefix=f'{data_dir_prefix_path}/{split}'):
        if '.tfrecord' in blob.name:
            train_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))
            
    print(f"train_files: {train_files}")

    train_dataset = tf.data.TFRecordDataset(train_files)
    # train_dataset.repeat().batch(batch_size)
    train_dataset = train_dataset.map(data_utils.parse_tfrecord) #, num_parallel_calls=tf.data.AUTOTUNE)
    
    return train_dataset

def train_perarm(
    agent,
    global_dim: int,
    per_arm_dim: int, 
    num_iterations: int,
    steps_per_loop: int,
    num_eval_steps: int,
    log_dir: str,
    model_dir: str,
    root_dir: str,
    batch_size: int,
    eval_batch_size: int,
    bucket_name: str,
    data_dir_prefix_path: str,
    # split: str,
    _trajectory_fn = None,
    # _run_bandit_eval_fn = None,
    log_interval: int = 1,
    chkpt_interval: int = 1,
    async_steps_per_loop = 1,
    resume_training_loops = False,
    additional_metrics: Optional[List[TFStepMetric]] = None,
    use_gpu = False,
    use_tpu = False,
) -> Dict[str, List[float]]:
    
    # GPU - All variables and Agents need to be created under strategy.scope()
    distribution_strategy = strategy_utils.get_strategy(tpu=use_tpu, use_gpu=use_gpu)
    print(f"distribution_strategy: {distribution_strategy}")
    
    # ====================================================
    # train dataset
    # ====================================================
    train_dataset = _get_train_dataset(
        bucket_name=bucket_name, 
        data_dir_prefix_path=data_dir_prefix_path, 
        split="train",
        batch_size = batch_size
        
    )
    # train_dataset.cache()
    # train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    # train_dataset = distribution_strategy.experimental_distribute_dataset(train_dataset)
    # train_ds_iterator = iter(train_dataset)
    train_ds_iterator = iter(train_dataset.repeat(num_iterations).batch(batch_size))
    print(f"train_ds_iterator: {train_ds_iterator}")
    
#     val_dataset = _get_train_dataset(
#         bucket_name=bucket_name, 
#         data_dir_prefix_path=data_dir_prefix_path, 
#         split="val"
#     )
#     eval_ds = val_dataset.batch(eval_batch_size) #.repeat(2)
    
#     if num_eval_steps > 0:
#         eval_ds = eval_ds.take(num_eval_steps)

#     # ====================================================
#     # TB summary writer
#     # ====================================================
#     print(f"log_dir: {log_dir}")
    
#     train_summary_writer = tf.compat.v2.summary.create_file_writer(
#         log_dir, flush_millis=10 * 1000
#     )
#     train_summary_writer.set_as_default()
    
    # ====================================================
    # metrics
    # ====================================================
    step_metric = tf_metrics.EnvironmentSteps()
    metrics = [
        # tf_metrics.NumberOfEpisodes(),
        # tf_metrics.AverageEpisodeLengthMetric(batch_size=batch_size),
        tf_metrics.AverageReturnMetric(batch_size=batch_size)
    ]
    if additional_metrics:
        metrics += additional_metrics

    metric_results = defaultdict(list)
    
    # ====================================================
    # chkpt and saver
    # ====================================================
    # get checkpoint manager
    CHKPOINT_DIR = f"{root_dir}/chkpoint"
    print(f"setting checkpoint_manager: {CHKPOINT_DIR}")
    
    checkpoint_manager = restore_and_get_checkpoint_manager(
        root_dir=CHKPOINT_DIR, 
        agent=agent, 
        metrics=metrics, 
        step_metric=step_metric
    )
    
    train_step_counter = tf.compat.v1.train.get_or_create_global_step()
    
    saver = policy_saver.PolicySaver(
        agent.policy, 
        train_step=train_step_counter
    )
    
    if resume_training_loops:
        train_step_count_per_loop = (
            steps_per_loop * batch_size * async_steps_per_loop
        )
        last_checkpointed_step = step_metric.result().numpy()
        if last_checkpointed_step % train_step_count_per_loop != 0:
            raise ValueError(
                'Last checkpointed step is expected to be a multiple of '
                'steps_per_loop * batch_size * async_steps_per_loop, but found '
                f'otherwise: last checkpointed step: {last_checkpointed_step}, '
                f'steps_per_loop: {steps_per_loop}, batch_size: '
                f'{batch_size}, async_steps_per_loop: '
                f'{async_steps_per_loop}'
            )
        starting_loop = last_checkpointed_step // train_step_count_per_loop
    else:
        starting_loop = 0
        
#     # ====================================================
#     # Evaluate the agent's policy once before training
#     # ====================================================
#     # Reset the train step
#     agent.train_step_counter.assign(0)

#     pre_policy_tf = py_tf_eager_policy.PyTFEagerPolicy(
#         agent.policy, use_tf_function=True
#     )

#     print(f"evaluating pre-trained Agent...")
#     start_time = time.time()

#     pre_val_loss, pre_preds, pre_tr_rewards = _run_bandit_eval_fn(
#         policy = pre_policy_tf,
#         data = eval_ds,
#         eval_batch_size = eval_batch_size,
#         per_arm_dim = per_arm_dim,
#         global_dim = global_dim
#     )
#     runtime_mins = int((time.time() - start_time) / 60)
#     print(f"pre-train val_loss: {pre_val_loss}")
#     print(f"pre-train eval runtime : {runtime_mins}")
    
    # ====================================================
    # train loop
    # ====================================================  
    print(f"starting_loop: {starting_loop}")

    # start the timer and training
    print(f"starting train loop...")
    start_time = time.time()
    list_o_loss = []
        
    for i in range(num_iterations):

        data = next(train_ds_iterator)
        trajectories = _trajectory_fn(data)

        step = agent.train_step_counter.numpy()
        loss = agent.train(experience=trajectories)
        list_o_loss.append(loss.loss.numpy())
        
        train_utils._export_metrics_and_summaries(
            step=i, 
            metrics=metrics
        )

        # print 
        if step % log_interval == 0:
            print(
                'step = {0}: loss = {1}'.format(
                    step, round(loss.loss.numpy(), 2)
                )
            )

        if i > 0 and i % chkpt_interval == 0:
            saver.save(os.path.join(CHKPOINT_DIR, 'policy_%d' % step_metric.result()))
            print(f"saved policy to: {CHKPOINT_DIR}")
            
    runtime_mins = int((time.time() - start_time) / 60)
    print(f"runtime_mins: {runtime_mins}")

    saver.save(model_dir)
    print(f"saved trained policy to: {model_dir}")
    
#     # ====================================================
#     # Evaluate the agent's policy once after training
#     # ====================================================
#     print(f"Load trained policy & evaluate...")
#     print(f"load policy from model_dir: {model_dir}")
    
#     # post_policy_tf = py_tf_eager_policy.PyTFEagerPolicy(
#     #     agent.policy, use_tf_function=True
#     # )
#     trained_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
#         model_dir, 
#         load_specs_from_pbtxt=True
#     )
#     print(f"trained_policy: {trained_policy}")
#     print(f"evaluating trained Policy...")
    
#     start_time = time.time()

#     val_loss, preds, tr_rewards = _run_bandit_eval_fn(
#         policy = trained_policy,
#         data = eval_ds,
#         eval_batch_size = eval_batch_size,
#         per_arm_dim = per_arm_dim,
#         global_dim = global_dim
#     )

#     runtime_mins = int((time.time() - start_time) / 60)
#     print(f"post-train val_loss: {val_loss}")
#     print(f"post-train eval runtime : {runtime_mins}")
    
    return list_o_loss, agent  # agent | val_loss

#, val_loss