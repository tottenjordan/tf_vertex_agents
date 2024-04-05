import os
import time
import logging
import functools
import numpy as np
import pickle as pkl
from pprint import pprint
from datetime import datetime
from collections import defaultdict
from typing import Callable, Dict, List, Optional, TypeVar
import collections

import tensorflow as tf

if tf.__version__[0] != "2":
    raise Exception("The trainer only runs with TensorFlow version 2.")

# TF-Agents
from tf_agents.utils import common
from tf_agents.specs import array_spec
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.metrics import tf_metrics
from tf_agents.bandits.metrics import tf_metrics as tf_bandit_metrics
from tf_agents.trajectories import trajectory
from tf_agents import trajectories
from tf_agents.metrics.tf_metric import TFStepMetric
from tf_agents.eval import metric_utils
from tf_agents.metrics import export_utils
from tf_agents.policies import policy_saver
from tf_agents.train.utils import strategy_utils
from tf_agents.utils import eager_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.bandits.policies import policy_utilities
from tf_agents.policies import py_tf_eager_policy

# logging
import logging
logging.disable(logging.WARNING)

from google.cloud import aiplatform
from google.cloud import storage

# this repo
# from . import eval_perarm
from src.utils import train_utils as train_utils
from src.data import data_config as data_config
from src.agents import agent_factory as agent_factory

# clients
storage_client = storage.Client(project=data_config.PROJECT_ID)

# ====================================================
# get train & val datasets
# ====================================================
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
options.threading.max_intra_op_parallelism = 1

def train(
    hparams: dict,
    experiment_name: str,
    experiment_run: str,
    num_epochs: int,
    log_dir: str,
    artifacts_dir: str,
    chkpoint_dir: str,
    tfrecord_file: str,
    log_interval: int = 10,
    # chkpt_interval: int = 100,
    use_gpu = False,
    use_tpu = False,
    # profiler = False,
    total_take: int = 0,
    total_skip: int = 0,
    cache_train_data = True,
    strategy: tf.distribute.Strategy = None,
    # train_summary_writer: Optional[tf.summary.SummaryWriter] = None,
    # additional_metrics: Optional[List[TFStepMetric]] = None,
) -> Dict[str, List[float]]:
    
    TF_GPU_THREAD_COUNT = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = f"{TF_GPU_THREAD_COUNT}"
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    aiplatform.init(
        project=data_config.PROJECT_ID,
        location="us-central1",
        experiment=experiment_name
    )
    print(f"hparams dict:")
    pprint(hparams)
    # ====================================================
    # helper functions
    # ====================================================
    # Mapping from feature name to serialized value
    _feature_description = {
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
        return tf.io.parse_single_example(raw_record, _feature_description)

    def _build_trajectory_from_tfrecord(
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
    # ====================================================
    # tensorspecs
    # ====================================================
    observation_spec = {
        'global': tf.TensorSpec([hparams['global_dim']], tf.float32),
        'per_arm': tf.TensorSpec([hparams['num_actions'], hparams['per_arm_dim']], tf.float32)
    }
    action_spec = tensor_spec.BoundedTensorSpec(
        shape=[], 
        dtype=tf.int32,
        minimum=tf.constant(0),            
        maximum=hparams['num_actions']-1,
        name="action_spec"
    )
    time_step_spec = ts.time_step_spec(observation_spec = observation_spec)
    reward_spec = {
        "reward": array_spec.ArraySpec(
            shape=[hparams['batch_size']], 
            dtype=np.float32, name="reward"
        )
    }
    reward_tensor_spec = train_utils.from_spec(reward_spec)
    # ====================================================
    # distribution strategy
    # ====================================================
    # GPU - All variables & Agents need to be created under strategy.scope()
    if strategy is None:
        distribution_strategy = strategy_utils.get_strategy(
            tpu=use_tpu, use_gpu=use_gpu
        )
    else:
        distribution_strategy = strategy
    NUM_REPLICAS = distribution_strategy.num_replicas_in_sync
    GLOBAL_BATCH_SIZE = int(hparams['batch_size']) * int(NUM_REPLICAS)

    tf.print(f'GLOBAL_BATCH_SIZE     : {GLOBAL_BATCH_SIZE}')
    tf.print(f"distribution_strategy : {distribution_strategy}")
    tf.print(f"NUM_REPLICAS          : {NUM_REPLICAS}")
    # ====================================================
    # metrics and summaries
    # ====================================================
    with distribution_strategy.scope():
        train_summary_writer = tf.compat.v2.summary.create_file_writer(
            f"{log_dir}", flush_millis=10 * 1000
        )
        train_summary_writer.set_as_default()
        
#     def _get_rewards(observation):
#         return observation.reward # target_movie_rating (?)

#     def optimal_reward(observation):
#         """
#         Outputs the maximum expected reward for every element in the batch
#         """
#         return tf.reduce_max(_get_rewards(observation), axis=1)
#     optimal_reward_fn = functools.partial(optimal_reward)

#     regret_metric = tf_bandit_metrics.RegretMetric(optimal_reward_fn)

    step_metric = tf_metrics.EnvironmentSteps()
    metrics = [
        tf_metrics.AverageReturnMetric(batch_size=hparams['batch_size']),
        # regret_metric
    ]
    # metric_results = collections.defaultdict(list)
    # ====================================================
    # get agent
    # ====================================================
    with distribution_strategy.scope():
        global_step = tf.compat.v1.train.get_or_create_global_step()
        agent = agent_factory.PerArmAgentFactory._get_agent(
            agent_type = hparams['agent_type'],
            network_type = hparams['network_type'],
            time_step_spec = time_step_spec,
            action_spec = action_spec,
            observation_spec=observation_spec,
            global_layers = hparams['global_layers'],
            arm_layers = hparams['per_arm_layers'],
            common_layers = hparams['common_layers'],
            agent_alpha = hparams['agent_alpha'],
            learning_rate = hparams['learning_rate'],
            epsilon = hparams['epsilon'],
            train_step_counter = global_step,
            output_dim = hparams['encoding_dim'],
            eps_phase_steps = hparams['eps_phase_steps'],
            summarize_grads_and_vars = hparams['summarize_grads_and_vars'],
            debug_summaries = hparams['debug_summaries'],
        )
        agent.initialize()
    # ====================================================
    # checkpointer
    # ====================================================
    checkpoint_manager = train_utils.restore_and_get_checkpoint_manager(
        root_dir=chkpoint_dir, 
        agent=agent, 
        metrics=metrics, 
        step_metric=step_metric
    )
    tf.print(f"set checkpoint_manager: {chkpoint_dir}")
    saver = policy_saver.PolicySaver(
        agent.policy, 
        train_step=global_step
    )
    tf.print(f"set saver: {saver}")
    # ====================================================
    # get dataset
    # ====================================================
    _raw_dataset = tf.data.TFRecordDataset([tfrecord_file])
    _parsed_dataset = _raw_dataset.map(_parse_record).prefetch(
        tf.data.experimental.AUTOTUNE
    )
    if total_skip > 0:
        _parsed_dataset = _parsed_dataset.skip(total_skip)
        tf.print(f"setting dataset total_skip: {total_skip}")
        
    if total_take > 0:
        _parsed_dataset = _parsed_dataset.take(count=total_take)
        tf.print(f"setting dataset total_take: {total_take}")

    # ====================================================
    # optimized train step
    # ====================================================
    @common.function(autograph=False)
    def _train_step_fn(trajectories):

        def replicated_train_step(experience):
            return agent.train(experience).loss

        per_replica_losses = distribution_strategy.run(
            replicated_train_step, 
            args=(trajectories,)
        )
        return distribution_strategy.reduce(
            tf.distribute.ReduceOp.MEAN, 
            per_replica_losses, # loss, 
            axis=None
        )
    # ====================================================
    # train loop
    # ====================================================
    loss_values = []
    with distribution_strategy.scope():
        
        tf.print(f"starting train loop...")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            
            tf.print(f"epoch: {epoch+1}")
            
            for parsed_record in _parsed_dataset:
                _parsed_trajectories = _build_trajectory_from_tfrecord(
                    parsed_record, hparams['batch_size'], hparams['num_actions']
                )
                step = agent.train_step_counter.numpy()
                loss = _train_step_fn(_parsed_trajectories)
                loss_values.append(loss.numpy())

                train_utils._export_metrics_and_summaries(
                    step=step,
                    metrics=metrics
                )
                
                if step % log_interval == 0:
                    tf.print(
                        'step = {0}: loss = {1}'.format(
                            step, round(loss.numpy(), 2)
                        )
                    )
                # if step > 0 and step % chkpt_interval == 0:
                #     checkpoint_manager.save(global_step)

        runtime_mins = int((time.time() - start_time) / 60)
        checkpoint_manager.save(global_step)
        saver.save(artifacts_dir)
        tf.print(f"train runtime_mins          : {runtime_mins}")
        tf.print(f"saved to checkpoint_manager : {chkpoint_dir}")
        tf.print(f"saved trained policy to     : {artifacts_dir}")
    
    return loss_values, agent