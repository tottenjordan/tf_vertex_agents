# ====================================================
# train loop 1
# ====================================================
import os
import time
import logging
import functools
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

if tf.__version__[0] != "2":
    raise Exception("The trainer only runs with TensorFlow version 2.")

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
from tf_agents.utils import eager_utils

# logging
import logging
logging.disable(logging.WARNING)

from google.cloud import aiplatform as vertex_ai
from google.cloud import storage

# this repo
from src.utils import train_utils as train_utils
from src.data import data_config as data_config

# clients
storage_client = storage.Client(project=data_config.PROJECT_ID)

# ====================================================
# get train & val datasets
# ====================================================
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
options.threading.max_intra_op_parallelism = 1 # TODO
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

def train_perarm(
    agent,
    reward_spec,
    # epsilon,
    embs,
    train_files,
    hparams: dict,
    global_dim: int,
    per_arm_dim: int, 
    num_iterations: int,
    num_epochs: int,
    steps_per_loop: int,
    log_dir: str,
    model_dir: str,
    chkpoint_dir: str,
    # batch_size: int,
    # bucket_name: str,
    # data_dir_prefix_path: str,
    # _trajectory_fn = None,
    log_interval: int = 10,
    chkpt_interval: int = 100,
    additional_metrics: Optional[List[TFStepMetric]] = None,
    use_gpu = False,
    use_tpu = False,
    profiler = False,
    global_step = None,
    # num_replicas = 1,
    # cache_train_data = True,
    saver = None,
    strategy: tf.distribute.Strategy = None,
    train_summary_writer: Optional[tf.summary.SummaryWriter] = None,
    use_tf_functions: bool = True,
    # is_testing: bool = False,
) -> Dict[str, List[float]]:
    
    # # GPU - All variables & Agents need to be created under strategy.scope()
    if strategy is None:
        distribution_strategy = strategy_utils.get_strategy(
            tpu=use_tpu, use_gpu=use_gpu
        )
    else:
        distribution_strategy = strategy
    tf.print(f"distribution_strategy: {distribution_strategy}")

    # ====================================================
    # metrics
    # ====================================================
    step_metric = tf_metrics.EnvironmentSteps()
    if not global_step:
        global_step = tf.compat.v1.train.get_or_create_global_step()

    metrics = [tf_metrics.AverageReturnMetric(batch_size=hparams['batch_size'])]
    if additional_metrics:
        metrics += additional_metrics

    # ====================================================
    # chkpt and saver
    # ====================================================
    tf.print("Inpsecting agent policy from train_peram file...")
    tf.print(f"agent.policy: {agent.policy}")

    # get checkpoint manager
    tf.print(f"setting checkpoint_manager: {chkpoint_dir}")
    checkpoint_manager = train_utils.restore_and_get_checkpoint_manager(
        root_dir=chkpoint_dir, 
        agent=agent, 
        metrics=metrics, 
        step_metric=step_metric
    )
    tf.print(f"agent.train_step_counter: {agent.train_step_counter.value().numpy()}")

    # ====================================================
    # saver
    # ====================================================
    saver = policy_saver.PolicySaver(
        agent.policy,
        train_step=global_step,
        # batch_size=None,
    )
    # ====================================================
    # TB summary writer
    # ====================================================
    if not train_summary_writer:
        train_summary_writer = tf.compat.v2.summary.create_file_writer(
            f"{log_dir}", flush_millis=10 * 1000
        )
    train_summary_writer.set_as_default()
    # ====================================================
    # train dataset
    # ====================================================
    process_example_fn = functools.partial(
        train_utils.example_proto_to_trajectory,
    )
    process_trajectory_fn = functools.partial(
        train_utils._trajectory_fn,
        hparams=hparams,
        embs=embs
    )
    train_dataset = train_utils.create_tfrecord_ds(
        tf.io.gfile.glob(train_files),
        # num_shards = 10,
        process_example_fn=process_example_fn,
        process_trajectory_fn=process_trajectory_fn,
        batch_size=hparams['batch_size'],
        repeat = True,
        drop_remainder = False
    )
    train_dataset_iterator = iter(train_dataset)
    
    # ====================================================
    # train step function
    # ====================================================
    if use_tf_functions:
        agent.train = common.function(agent.train)
        
    def train_step():
        experience = next(train_dataset_iterator)
        return agent.train(experience).loss
    
    # def train_step(experience):
    #     return agent.train(experience).loss
    
    if use_tf_functions:
        train_step = common.function(train_step)

    # start the timer and training
    with tf.compat.v2.summary.record_if(
        lambda: tf.math.equal(global_step % log_interval, 0)
    ):

        tf.print(f"starting train loop...")
        list_o_loss = []
        timed_at_step = global_step.numpy()
        time_acc = 0

        # if profiler:
            # tf.profiler.experimental.start(log_dir)

        start_train = time.time()

        for epoch in tf.range(num_epochs):
            epoch_count = epoch+1
            tf.print(f"\nstarting epoch: {epoch_count}")
            start_epoch = time.time()

            for i in tf.range(num_iterations):
                step = agent.train_step_counter.numpy()

                # experience = next(train_dataset_iterator)
                # loss = train_step(experience)

                loss = train_step()

                list_o_loss.append(loss.numpy())

                if step % log_interval == 0:
                    tf.print(f'step = {step}: loss = {round(loss.numpy(), 2)}')

                # train_utils._export_metrics_and_summaries(
                #     step=step, metrics=metrics
                # )

            epoch_mins = int((time.time() - start_epoch) / 60)
            tf.print(f"epoch_mins: {epoch_mins}\n")

            time_acc += time.time() - start_train
            steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
            tf.print(f"{round(steps_per_sec,3)} steps/sec")
            time_acc = 0

    # if profiler:
    #     tf.profiler.experimental.stop()

    runtime_mins = int((time.time() - start_train) / 60)
    tf.print(f"runtime_mins: {runtime_mins}")

    saver.save(model_dir)
    tf.print(f"saved trained policy to: {model_dir}")

    checkpoint_manager.save(global_step)
    print(f"saved to checkpoint_manager: {chkpoint_dir}")
    
    return list_o_loss, agent
#     # ==== Distributed ========#
    
#     # ====================================================
#     # distributed train setp function
#     # ====================================================
#     # (Optional) Optimize by wrapping some of the 
#     # code in a graph using TF function.
#     # Big perfromance boost right here
    
#     # tf.print('wrapping agent.train in tf-function')
#     # agent.train = common.function(agent.train)
    
#     # data = next(train_ds_iterator)
#     # def _train_step_fn(iterator):
#     # data = eager_utils.get_next(iterator)
    
#     # @tf.function # TODO: replace numpy with TF for perf boost    
#     @common.function(autograph=False)
#     def _dist_train_step_fn(train_ds_iterator):
        
#         def replicated_train_step(experience):
#             return agent.train(experience).loss
        
#         # trajectories = _trajectory_fn(data)
#         trajectories = next(train_ds_iterator)

#         per_replica_losses = distribution_strategy.run(
#             replicated_train_step, 
#             args=(trajectories,)
#         )

#         # return agent.train(experience=trajectories).loss
#         return distribution_strategy.reduce(
#             tf.distribute.ReduceOp.MEAN, 
#             per_replica_losses, # loss, 
#             axis=None
#         )
    
#     # start the timer and training
#     tf.print(f"starting train loop...")
#     list_o_loss = []
#     # ====================================================
#     # profiler - train loop
#     # ====================================================
#     if profiler:
#         tf.profiler.experimental.start(log_dir)
    
#     start_time = time.time()
#     with distribution_strategy.scope():
        
#         # dist_dataset = distribution_strategy.experimental_distribute_dataset(train_dataset)
#         # train_ds_iterator = iter(dist_dataset)
        
#         for epoch in tf.range(num_epochs):
#             tf.print(f"epoch: {epoch+1}")
                
#             for i in tf.range(num_iterations):
#                 step = agent.train_step_counter.numpy()
                
#                 # data = next(train_ds_iterator)
#                 # loss = _dist_train_step_fn(data)
                
#                 trajectories = next(train_ds_iterator)
#                 loss = agent.train(trajectories).loss
                
#                 list_o_loss.append(loss.numpy())
                
#                 train_utils._export_metrics_and_summaries(
#                     step=step,
#                     metrics=metrics
#                 )

#                 if step % log_interval == 0:
#                     tf.print(
#                         'step = {0}: loss = {1}'.format(
#                             step, round(loss.numpy(), 2)
#                         )
#                     )
#         if profiler:
#             tf.profiler.experimental.stop()

#     runtime_mins = int((time.time() - start_time) / 60)
#     tf.print(f"runtime_mins: {runtime_mins}")
    
#     saver.save(model_dir)
#     tf.print(f"saved trained policy to: {model_dir}")
    
#     checkpoint_manager.save(global_step)
#     print(f"saved to checkpoint_manager: {chkpoint_dir}")
    
#     # ====================================================
#     # non-profiler - train loop
#     # ====================================================
        # with tf.profiler.experimental.Trace(
        #     "tr_step", step_num=step, _r=1 # step.numpy()
        # ):

#     if not profiler:
        
#         start_time = time.time()
        
#         for i in tf.range(num_epochs):
#             tf.print(f"epoch: {i+1}")
        
#             with distribution_strategy.scope():

#                 for i in tf.range(num_iterations):

#                     step = agent.train_step_counter
#                     data = next(train_ds_iterator)
#                     loss = _train_step_fn(data)
#                     list_o_loss.append(loss.numpy())

#                     train_utils._export_metrics_and_summaries(
#                         step=step.numpy(), 
#                         metrics=metrics
#                     )
#                     if step % log_interval == 0:
#                         tf.print(
#                             'step = {0}: loss = {1}'.format(
#                                 step.numpy(), round(loss.numpy(), 2)
#                             )
#                         )
#                     if i > 0 and i % chkpt_interval == 0:
#                         checkpoint_manager.save(global_step)

#         runtime_mins = int((time.time() - start_time) / 60)
#         tf.print(f"runtime_mins: {runtime_mins}")

#     saver.save(model_dir)
#     tf.print(f"saved trained policy to: {model_dir}")
#     checkpoint_manager.save(global_step)
#     tf.print(f"saved trained policy to: {chkpoint_dir}")
    
    # return list_o_loss, agent  # agent | val_loss