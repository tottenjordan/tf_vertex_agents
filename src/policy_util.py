# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The utility module for reinforcement learning policy."""
import collections
from typing import Callable, Dict, List, Optional, TypeVar
import argparse
import functools
import json
import logging
import os
import sys
from typing import List, Union
import time
import random
import string

import logging
logging.disable(logging.WARNING)

import tensorflow as tf

from tf_agents.agents import TFAgent
from tf_agents.bandits.agents.examples.v2 import trainer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import TFEnvironment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.metrics.tf_metric import TFStepMetric
from tf_agents.policies import policy_saver

from tf_agents.metrics import export_utils
from tf_agents.bandits.replay_buffers import bandit_replay_buffer
import time

# this repo
from . import train_utils

# import traceback
# from google.cloud.aiplatform.training_utils import cloud_profiler

def _get_training_loop(
    driver, replay_buffer, agent, steps, async_steps_per_loop, log_interval=10
):
    """Returns a `tf.function` that runs the driver and training loops.

    Args:
    driver: an instance of `Driver`.
    replay_buffer: an instance of `ReplayBuffer`.
    agent: an instance of `TFAgent`.
    steps: an integer indicating how many driver steps should be executed and
      presented to the trainer during each training loop.
    async_steps_per_loop: an integer. In each training loop, the driver runs
      this many times, and then the agent gets asynchronously trained over this
      many batches sampled from the replay buffer.
    """

    def _export_metrics_and_summaries(step, metrics):
        """Exports metrics and tf summaries."""
        metric_utils.log_metrics(metrics)
        export_utils.export_metrics(step=step, metrics=metrics)
        for metric in metrics:
            metric.tf_summaries(train_step=step)

    def training_loop(train_step, metrics):
        """Returns a function that runs a single training loop and logs metrics."""
        for batch_id in range(async_steps_per_loop):
            driver.run()
            _export_metrics_and_summaries(
                step=train_step * async_steps_per_loop + batch_id, metrics=metrics
            )
        batch_size = driver.env.batch_size
        dataset_it = iter(
            replay_buffer.as_dataset(
                sample_batch_size=batch_size,
                num_steps=steps,
                single_deterministic_pass=True,
            )
        )
        for batch_id in range(async_steps_per_loop):
            experience, unused_buffer_info = dataset_it.get_next()
            train_utils.set_expected_shape(experience, steps)
            loss_info = agent.train(experience)
            export_utils.export_metrics(
                step=train_step * async_steps_per_loop + batch_id,
                metrics=[],
                loss_info=loss_info,
            )
            if train_step % log_interval == 0:
                print(
                    f'step = {train_step}: train loss = {round(loss_info.loss.numpy(), 4)}'
                )

        replay_buffer.clear()

    return training_loop

T = TypeVar("T")

def train(
    agent: TFAgent
    , environment: TFEnvironment
    , training_loops: int
    , steps_per_loop: int
    , log_dir: str
    , chkpt_dir: str
    , profiler: bool
    , chkpt_interval: int = 25
    , additional_metrics: Optional[List[TFStepMetric]] = None
    , training_data_spec_transformation_fn: Optional[Callable[[T],T]] = None
    , run_hyperparameter_tuning: bool = False
    , root_dir: Optional[str] = None
    , artifacts_dir: Optional[str] = None
    , model_dir: Optional[str] = None
    , train_summary_writer: Optional[tf.summary.SummaryWriter] = None
    , global_step = None
) -> Dict[str, List[float]]:
    """
    Performs `training_loops` iterations of training on the agent's policy.

    Uses the `environment` as the problem formulation and source of immediate
    feedback and the agent's algorithm, to perform `training-loops` iterations
    of on-policy training on the policy. Has hyperparameter mode and regular
    training mode.
    If one or more baseline_reward_fns are provided, the regret is computed
    against each one of them. Here is example baseline_reward_fn:
    def baseline_reward_fn(observation, per_action_reward_fns):
     rewards = ... # compute reward for each arm
     optimal_action_reward = ... # take the maximum reward
     return optimal_action_reward

    Args:
      agent: An instance of `TFAgent`.
      environment: An instance of `TFEnvironment`.
      training_loops: An integer indicating how many training loops should be run.
      steps_per_loop: An integer indicating how many driver steps should be
        executed and presented to the trainer during each training loop.
      additional_metrics: Optional; list of metric objects to log, in addition to
        default metrics `NumberOfEpisodes`, `AverageReturnMetric`, and
        `AverageEpisodeLengthMetric`.
      training_data_spec_transformation_fn: Optional; function that transforms
        the data items before they get to the replay buffer.
      run_hyperparameter_tuning: Optional; whether this training logic is
        executed for the purpose of hyperparameter tuning. If so, then it does
        not save model artifacts.
      root_dir: Optional; path to the directory where training artifacts are
        written; usually used for a default or auto-generated location. Do not
        specify this argument if using hyperparameter tuning instead of training.
      artifacts_dir: Optional; path to an extra directory where training
        artifacts are written; usually used for a mutually agreed location from
        which artifacts will be loaded. Do not specify this argument if using
        hyperparameter tuning instead of training.

    Returns:
      A dict mapping metric names (eg. "AverageReturnMetric") to a list of
      intermediate metric values over `training_loops` iterations of training.
    """
    # ====================================================
    # TB summary writer
    # ====================================================
    if train_summary_writer:
        train_summary_writer.set_as_default()

    logging.info(f" log_dir: {log_dir}")
    # ====================================================
    # get data spec
    # ====================================================
    if run_hyperparameter_tuning and not (root_dir is None and artifacts_dir is None):
        raise ValueError(
            "Do not specify `root_dir` or `artifacts_dir` when" +
            " running hyperparameter tuning."
        )

    if training_data_spec_transformation_fn is None:
        data_spec = agent.policy.trajectory_spec
    else:
        data_spec = training_data_spec_transformation_fn(
            agent.policy.trajectory_spec
        )

    # ====================================================
    # define replay buffer
    # ====================================================
    replay_buffer = trainer._get_replay_buffer(
        data_spec = data_spec
        , batch_size = environment.batch_size
        , steps_per_loop = steps_per_loop
        , async_steps_per_loop = 1
    )
    # ====================================================
    # metrics
    # ====================================================
    # `step_metric` records the number of individual rounds of bandit interaction;
    # that is, (number of trajectories) * batch_size.
    step_metric = tf_metrics.EnvironmentSteps()
    metrics = [
        tf_metrics.NumberOfEpisodes()
        , tf_metrics.EnvironmentSteps()
        , tf_metrics.AverageEpisodeLengthMetric(batch_size=environment.batch_size)
    ]
    if additional_metrics:
        metrics += additional_metrics

    if isinstance(environment.reward_spec(), dict):
        metrics += [
            tf_metrics.AverageReturnMultiMetric(
                reward_spec=environment.reward_spec()
                , batch_size=environment.batch_size
            )
        ]
    else:
        metrics += [
            tf_metrics.AverageReturnMetric(batch_size=environment.batch_size)
        ]
    # Store intermediate metric results, indexed by metric names.
    metric_results = collections.defaultdict(list)

    # ====================================================
    # Policy checkpoints
    # ====================================================
    if 'AIP_CHECKPOINT_DIR' in os.environ:
        CHKPOINT_DIR=os.environ['AIP_CHECKPOINT_DIR']
        logging.info(f'AIP_CHECKPOINT_DIR: {CHKPOINT_DIR}')
    else:
        CHKPOINT_DIR=chkpt_dir

    logging.info(f"setting checkpoint_manager: {CHKPOINT_DIR}")
    checkpoint_manager = train_utils.restore_and_get_checkpoint_manager(
        root_dir=CHKPOINT_DIR, 
        agent=agent, 
        metrics=metrics, 
        step_metric=step_metric
    )

    # ====================================================
    # Driver
    # ====================================================
    if training_data_spec_transformation_fn is not None:
        add_batch_fn = lambda data: replay_buffer.add_batch(
            training_data_spec_transformation_fn(data)
        )
    else:
        add_batch_fn = replay_buffer.add_batch

    observers = [add_batch_fn, step_metric] + metrics

    driver = dynamic_step_driver.DynamicStepDriver(
        env=environment
        , policy=agent.collect_policy
        , num_steps=steps_per_loop * environment.batch_size
        , observers=observers
    )

    # ====================================================
    # training_loop
    # ====================================================
    if profiler:
        # start_profiling_step = 10
        # stop_profiling_step = 50
        profiler_options = tf.profiler.experimental.ProfilerOptions(
            host_tracer_level = 3
            , python_tracer_level = 1
            , device_tracer_level = 1
        )
        # logging.info(f'start_profiling_step : {start_profiling_step}')
        # logging.info(f'stop_profiling_step  : {stop_profiling_step}')
        logging.info(f'profiler_options     : {profiler_options}')
    
    training_loop = _get_training_loop(
        driver = driver
        , replay_buffer = replay_buffer
        , agent = agent
        , steps = steps_per_loop
        , async_steps_per_loop = 1
    )

    # train_step_counter = tf.compat.v1.train.get_or_create_global_step()

    # if not run_hyperparameter_tuning:
    #     saver = policy_saver.PolicySaver(
    #         agent.policy, 
    #         train_step=global_step.numpy()
    #     )
    #     tf.print(f"created saver: {saver}")
    # saver = policy_saver.PolicySaver(agent.policy)
        
    # ====================================================
    # profiler - train loop
    # ====================================================   
    if profiler:
        start_time = time.time()
        tf.profiler.experimental.start(log_dir)
        
        for train_step in range(training_loops):
                
            with tf.profiler.experimental.Trace(
                "tr_step", step_num=train_step, _r=1
            ):
                # training loop
                training_loop(
                    train_step = train_step
                    , metrics = metrics
                )

#                 # log tensorboard
#                 for metric in metrics:
#                     metric.tf_summaries(
#                         train_step=train_step
#                         , step_metrics=metrics[:2]
#                     )

#                 metric_utils.log_metrics(metrics)

#                 for metric in metrics:
#                     metric.tf_summaries(train_step = step_metric.result())
#                     metric_results[type(metric).__name__].append(metric.result().numpy())

        tf.profiler.experimental.stop()
        runtime_mins = int((time.time() - start_time) / 60)
        tf.print(f"runtime_mins: {runtime_mins}")
        
        checkpoint_manager.save(global_step)
        tf.print(f"saved policy checkpoint to: {CHKPOINT_DIR}")
        
    # ====================================================
    # non-profiler - train loop
    # ====================================================
    if not profiler:
        start_time = time.time()
        
        for train_step in range(training_loops):
            
            # training loop
            training_loop(
                train_step = train_step
                , metrics = metrics
            )

            # log tensorboard
            for metric in metrics:
                metric.tf_summaries(
                    train_step=train_step
                    , step_metrics=metrics[:2]
                )

            metric_utils.log_metrics(metrics)

            for metric in metrics:
                metric.tf_summaries(train_step = step_metric.result())
                metric_results[type(metric).__name__].append(metric.result().numpy())
                
        runtime_mins = int((time.time() - start_time) / 60)
        tf.print(f"runtime_mins: {runtime_mins}")
        
        checkpoint_manager.save(global_step)
        tf.print(f"saved policy checkpoint to: {CHKPOINT_DIR}")

    return metric_results