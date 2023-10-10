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

os.environ['TF_GPU_THREAD_MODE']='gpu_private'
os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'

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
from google.cloud.aiplatform.training_utils import cloud_profiler
import traceback

# this repo
from src.per_arm_rl import train_utils
from src.per_arm_rl import data_utils
from src.per_arm_rl import data_config

if tf.__version__[0] != "2":
    raise Exception("The trainer only runs with TensorFlow version 2.")

PER_ARM = True  # Use the non-per-arm version of the MovieLens environment.

# clients
project_number='hybrid-vertex' # TODO: param
storage_client = storage.Client(project=project_number)

# ====================================================
# get train & val datasets
# ====================================================
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
options.threading.max_intra_op_parallelism = 1 # TODO

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
    profiler = False,
    global_step = None,
    total_train_take: int = 10000,
    num_replicas = 1,
    cache_train_data = True,
    train_summary_writer: Optional[tf.summary.SummaryWriter] = None,
) -> Dict[str, List[float]]:
    
    if train_summary_writer:
        train_summary_writer.set_as_default()
    
    # # GPU - All variables & Agents need to be created under strategy.scope()
    # distribution_strategy = strategy_utils.get_strategy(
    #     tpu=use_tpu, use_gpu=use_gpu
    # )
    # print(f"distribution_strategy: {distribution_strategy}")
    
    # ====================================================
    # train dataset
    # ====================================================
    train_dataset = train_utils._get_train_dataset(
        bucket_name=bucket_name, 
        data_dir_prefix_path=data_dir_prefix_path, 
        split="train",
        total_take=total_train_take,
        batch_size = batch_size,
        num_replicas = num_replicas,
        cache = cache_train_data,
    )
    # train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    # train_dataset = train_dataset.cache()
    # train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    # train_dataset = distribution_strategy.experimental_distribute_dataset(train_dataset)
    train_ds_iterator = iter(train_dataset)
    print(f"train_ds_iterator: {train_ds_iterator}")
    
#     val_dataset = _get_train_dataset(
#         bucket_name=bucket_name, 
#         data_dir_prefix_path=data_dir_prefix_path, 
#         split="val"
#     )
#     eval_ds = val_dataset.batch(eval_batch_size) #.repeat(2)
    
#     if num_eval_steps > 0:
#         eval_ds = eval_ds.take(num_eval_steps)
    
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
    
    checkpoint_manager = train_utils.restore_and_get_checkpoint_manager(
        root_dir=CHKPOINT_DIR, 
        agent=agent, 
        metrics=metrics, 
        step_metric=step_metric
    )
    
    # train_step_counter = tf.compat.v1.train.get_or_create_global_step()
    
    saver = policy_saver.PolicySaver(
        agent.policy, 
        train_step=global_step
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

    # ====================================================
    # train setp function
    # ====================================================
    # @tf.function # TODO: replace numpy with TF for perf boost
    def _train_step_fn():
        
        data = next(train_ds_iterator)
        trajectories = _trajectory_fn(data)
        loss = agent.train(experience=trajectories)
        
        return loss #, step
    
    # (Optional) Optimize by wrapping some of the 
    # code in a graph using TF function.
    # Big perfromance boost right here
    print('wrapping agent.train in tf-function')
    agent.train = common.function(agent.train)
    
    print(f"starting_loop: {starting_loop}")

    # start the timer and training
    print(f"starting train loop...")
    list_o_loss = []
    
    # ====================================================
    # profiler - train loop
    # ====================================================    
    if profiler:
        start_time = time.time()
        tf.profiler.experimental.start(log_dir)
        for i in range(num_iterations):

            step = agent.train_step_counter.numpy()

            with tf.profiler.experimental.Trace(
                "tr_step", step_num=step, _r=1
            ):
                loss = _train_step_fn()

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
                saver.save(
                    os.path.join(
                        CHKPOINT_DIR, 
                        'policy_%d' % step_metric.result()
                    )
                )
                print(f"saved policy to: {CHKPOINT_DIR}")

        tf.profiler.experimental.stop()
        runtime_mins = int((time.time() - start_time) / 60)
        print(f"runtime_mins: {runtime_mins}")
    # ====================================================
    # non-profiler - train loop
    # ====================================================
    if not profiler:
        
        start_time = time.time()
        for i in range(num_iterations):

            step = agent.train_step_counter.numpy()
            loss = _train_step_fn()
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
                saver.save(
                    os.path.join(
                        CHKPOINT_DIR, 
                        'policy_%d' % step_metric.result()
                    )
                )
                print(f"saved policy to: {CHKPOINT_DIR}")
            
        runtime_mins = int((time.time() - start_time) / 60)
        print(f"runtime_mins: {runtime_mins}")

    saver.save(model_dir)
    print(f"saved trained policy to: {model_dir}")
    
    return list_o_loss, agent  # agent | val_loss