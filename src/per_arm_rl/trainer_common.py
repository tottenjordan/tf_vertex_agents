

# ====================================================
# train loop 1
# ====================================================
import os
import time
import logging
from collections import defaultdict
from typing import Callable, Dict, List, Optional, TypeVar

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

# this repo
from src.per_arm_rl import train_utils

# # replay buffer function
def _get_replay_buffer(
    data_spec, batch_size, steps_per_loop, async_steps_per_loop
):
    """Return a `TFUniformReplayBuffer` for the given `agent`."""
    # return bandit_replay_buffer.BanditReplayBuffer(
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=data_spec,
        batch_size=batch_size,
        max_length=steps_per_loop * async_steps_per_loop,
        # device='cpu:*',
    )

def set_expected_shape(experience, num_steps):
    """
    Sets expected shape.
    """

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
# Train loop
# ====================================================
def _get_training_loop(
    driver, 
    replay_buffer, 
    agent, 
    steps, 
    async_steps_per_loop,
    metrics,
    log_interval = 1,
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
    
    def training_loop(train_step, metrics, log_interval, metric_results,step_metric):
        """
        Returns a function that runs a single training loop and logs metrics.
        """
        
        start_step_time = time.time()
        
        for batch_id in range(async_steps_per_loop):
            driver.run()
            _export_metrics_and_summaries(
                step=train_step * async_steps_per_loop + batch_id, 
                metrics=metrics
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
            set_expected_shape(experience, steps)
            loss_info = agent.train(experience)
            
            step_time = int((time.time() - start_step_time) / 60)
            
            if log_interval and train_step % log_interval == 0:
                print(
                    'step = {0}: loss = {1}; execution time: {2}'.format(
                        train_step, round(loss_info.loss.numpy(), 2), step_time
                    )
                )
            # # tmp - TODO - uncomment and observe impact
            # export_utils.export_metrics(
            #     step=train_step * async_steps_per_loop + batch_id,
            #     metrics=metrics, #[],
            #     loss_info=loss_info,
            # )
            # metric_utils.log_metrics(metrics)
            # for metric in metrics:
            #     metric.tf_summaries(
            #         train_step=step_metric.result(),
            #         # step_metrics=metrics[:2]
            #     )
            #     metric_results[type(metric).__name__].append(metric.result().numpy())

        replay_buffer.clear()

    return training_loop

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
        logging.info('Restoring checkpoint from %s.', latest)
        checkpoint.restore(latest)
        logging.info('Successfully restored to step %s.', step_metric.result())
    else:
        logging.info(
            'Did not find a pre-existing checkpoint. Starting from scratch.'
        )
    return checkpoint_manager
# ====================================================
# train, eval per-arm bandit
# ====================================================
'''
metric_results, regret_values = train_perarm(
    num_iterations = num_iterations,
    steps_per_loop = steps_per_loop,
    start_profiling_step=1,
    stop_profiling_step=2,
    log_interval = 1,
    additional_metrics = [regret_metric] + metrics,
    profiling_log_dir=LOG_DIR
)

pprint(metric_results)
'''

def train_perarm(
    agent,
    # replay_buffer,
    # driver,
    environment,
    # regret_metric,
    # step_metric,
    num_iterations: int,
    steps_per_loop: int,
    log_dir: str,
    model_dir: str,
    root_dir: str,
    log_interval: int = 1,
    async_steps_per_loop = None,
    resume_training_loops = False,
    get_replay_buffer_fn = None,
    get_training_loop_fn = None,
    training_data_spec_transformation_fn = None,
    additional_metrics: Optional[List[TFStepMetric]] = None,
) -> Dict[str, List[float]]:
    
    # GPU - All variables and Agents need to be created under strategy.scope()
    use_gpu = True
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)
    
    # ====================================================
    # get data spec
    # ====================================================
    # TODO(b/127641485): create evaluation loop with configurable metrics.
    if training_data_spec_transformation_fn is None:
        data_spec = agent.policy.trajectory_spec
    else:
        data_spec = training_data_spec_transformation_fn(
            agent.policy.trajectory_spec
        )
    if async_steps_per_loop is None:
        async_steps_per_loop = 1
        
    logging.info(f"async_steps_per_loop: {async_steps_per_loop}")
    
    # ====================================================
    # define replay buffer
    # ====================================================
    if get_replay_buffer_fn is None:
        get_replay_buffer_fn = _get_replay_buffer
    
    replay_buffer = get_replay_buffer_fn(
        data_spec, environment.batch_size, steps_per_loop, async_steps_per_loop
    )
    logging.info(f" replay_buffer: {replay_buffer.name}")
    
    # ====================================================
    # TB summary writer
    # ====================================================
    logging.info(f" log_dir: {log_dir}")
    
    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        log_dir, flush_millis=10 * 1000
    )
    train_summary_writer.set_as_default()
    
    # ====================================================
    # metrics
    # ====================================================
    step_metric = tf_metrics.EnvironmentSteps()
    metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.AverageEpisodeLengthMetric(batch_size=environment.batch_size)
    ]
    if additional_metrics:
        metrics += additional_metrics
        
    if isinstance(environment.reward_spec(), dict):
        metrics += [
            tf_metrics.AverageReturnMultiMetric(
                reward_spec=environment.reward_spec(),
                batch_size=environment.batch_size
            )
        ]
    else:
        metrics += [
            tf_metrics.AverageReturnMetric(batch_size=environment.batch_size)
        ]

    # if isinstance(
    #     environment.reward_spec(), dict
    # ) or environment.reward_spec().shape != tf.TensorShape(()):
    #     metrics += [
    #         tf_metrics.AverageReturnMultiMetric(
    #             reward_spec=environment.reward_spec(),
    #             batch_size=environment.batch_size
    #         )
    #     ]
    # else:
    #     metrics += [
    #         tf_metrics.AverageReturnMetric(batch_size=environment.batch_size)
    #     ]
    
    # if not isinstance(environment.reward_spec(), dict):
    #     metrics += [
    #         tf_metrics.AverageReturnMetric(batch_size=environment.batch_size)
    #     ]

    # Store intermediate metric results, indexed by metric names.
    metric_results = defaultdict(list)
    
    # ====================================================
    # Driver
    # ====================================================
    if training_data_spec_transformation_fn is not None:
        def add_batch_fn(data): return replay_buffer.add_batch(
            training_data_spec_transformation_fn(data)
        ) 
        
    else:
        add_batch_fn = replay_buffer.add_batch

    observers = [add_batch_fn, step_metric] + metrics

    driver = dynamic_step_driver.DynamicStepDriver(
        env = environment
        , policy = agent.collect_policy
        , num_steps = steps_per_loop * environment.batch_size
        , observers = observers
    )

    # ====================================================
    # train loop
    # ====================================================
    
    # # get train loop function
    if get_training_loop_fn is None:
        get_training_loop_fn = _get_training_loop
    
    training_loop = get_training_loop_fn(
        driver = driver, 
        replay_buffer = replay_buffer, 
        agent = agent,
        steps = steps_per_loop,
        async_steps_per_loop = async_steps_per_loop,
        log_interval = log_interval,
        metrics = metrics
    )
    
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
        agent.policy, train_step=train_step_counter
    )
    
    if resume_training_loops:
        train_step_count_per_loop = (
            steps_per_loop * environment.batch_size * async_steps_per_loop
        )
        last_checkpointed_step = step_metric.result().numpy()
        if last_checkpointed_step % train_step_count_per_loop != 0:
            raise ValueError(
                'Last checkpointed step is expected to be a multiple of '
                'steps_per_loop * batch_size * async_steps_per_loop, but found '
                f'otherwise: last checkpointed step: {last_checkpointed_step}, '
                f'steps_per_loop: {steps_per_loop}, batch_size: '
                f'{environment.batch_size}, async_steps_per_loop: '
                f'{async_steps_per_loop}'
            )
        starting_loop = last_checkpointed_step // train_step_count_per_loop
    else:
        starting_loop = 0
    
    print(f"starting_loop: {starting_loop}")

    #start the timer and training
    print(f"starting train loop...")
    start_time = time.time()
    
    for i in range(starting_loop, num_iterations):
        training_loop(
            train_step=i, 
            metrics=metrics, 
            log_interval=log_interval,
            metric_results=metric_results,
            step_metric=step_metric
        )
        # log metrics
        metric_utils.log_metrics(metrics)
        for metric in metrics:
            metric.tf_summaries(
                train_step=step_metric.result(),
                # step_metrics=metrics[:2]
            )
            metric_results[type(metric).__name__].append(metric.result().numpy())
            
        checkpoint_manager.save()
        if i > 0 and i % 100 == 0:
            saver.save(os.path.join(CHKPOINT_DIR, 'policy_%d' % step_metric.result()))
            print(f"saved policy to: {CHKPOINT_DIR}")
    
    runtime_mins = int((time.time() - start_time) / 60)
    print(f"runtime_mins: {runtime_mins}")
    
    saver.save(model_dir)
    print(f"saved trained policy to: {model_dir}")
    
    return metric_results
