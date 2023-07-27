

# ====================================================
# train loop 1
# ====================================================
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

# this repo
from src.per_arm_rl import train_utils

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
    replay_buffer,
    driver,
    environment,
    regret_metric,
    num_iterations: int,
    steps_per_loop: int,
    log_dir: str,
    model_dir: str,
    log_interval: int = 1,
    additional_metrics: Optional[List[TFStepMetric]] = None,
) -> Dict[str, List[float]]:
    
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
    metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.AverageEpisodeLengthMetric(batch_size=environment.batch_size)
    ]

    if isinstance(environment.reward_spec(), dict):
        metrics += [
            tf_metrics.AverageReturnMultiMetric(
                reward_spec=environment.reward_spec(),
                batch_size=environment.batch_size)
        ]
    else:
        metrics += [
            tf_metrics.AverageReturnMetric(batch_size=environment.batch_size)
        ]
        
    if additional_metrics:
        metrics += additional_metrics

    # Store intermediate metric results, indexed by metric names.
    metric_results = defaultdict(list)

    # ====================================================
    # train loop
    # ====================================================
    
    # `step_metric` records the number of individual rounds of bandit interaction;
    # that is, (number of trajectories) * batch_size.
    step_metric = tf_metrics.EnvironmentSteps()
    
    @tf.function(autograph=False)
    def train_step():
        return agent.train(replay_buffer.gather_all())

    regret_values = []
    # log_interval = 1

    #start the timer and training
    print(f"starting train loop...")
    start_time = time.time()

    for step in range(num_iterations):
        
        start_step_time = time.time()
            
        driver.run()

        train_utils._export_metrics_and_summaries(
            step=step, metrics=metrics
        )
        # loss_info = agent.train(replay_buffer.gather_all())
        loss_info = train_step()

        replay_buffer.clear()
        regret_values.append(regret_metric.result())

        step_time = int((time.time() - start_step_time) / 60)
        if log_interval and step % log_interval == 0:
            print(
                'step = {0}: loss = {1}; execution time: {2}'.format(
                    step, round(loss_info.loss.numpy(), 2), step_time
                )
            )

        export_utils.export_metrics(
            step=step,
            metrics=metrics, #[],
            loss_info=loss_info
        )

        metric_utils.log_metrics(metrics)
        for metric in metrics:
            metric.tf_summaries(
                train_step=step_metric.result(),
                step_metrics=metrics[:2]
            )
            metric_results[type(metric).__name__].append(metric.result().numpy())

        # print(f"step: {step} completed in {step_time}")

    runtime_mins = int((time.time() - start_time) / 60)
    print(f"runtime_mins: {runtime_mins}")
    
    saver = policy_saver.PolicySaver(agent.policy)
    saver.save(model_dir)
    
    # return metric_results
    return metric_results, regret_values


# # ====================================================
# # train loop 2
# # ====================================================
# def _get_training_loop(driver, replay_buffer, agent, steps,
#                        async_steps_per_loop):
#     """
#     Returns a `tf.function` that runs the driver and training loops.

#     Args:
#       driver: an instance of `Driver`.
#       replay_buffer: an instance of `ReplayBuffer`.
#       agent: an instance of `TFAgent`.
#       steps: an integer indicating how many driver steps should be
#         executed and presented to the trainer during each training loop.
#       async_steps_per_loop: an integer. In each training loop, the driver runs
#         this many times, and then the agent gets asynchronously trained over this
#         many batches sampled from the replay buffer.
#     """

#     def _export_metrics_and_summaries(step, metrics):
#         """Exports metrics and tf summaries."""
#         metric_utils.log_metrics(metrics)
#         export_utils.export_metrics(step=step, metrics=metrics)
#         for metric in metrics:
#             metric.tf_summaries(train_step=step)

#     def training_loop(train_step, metrics):
#         """Returns a function that runs a single training loop and logs metrics."""
#         for batch_id in range(async_steps_per_loop):
#             driver.run()
#             _export_metrics_and_summaries(
#                 step=train_step * async_steps_per_loop + batch_id, metrics=metrics)
#         batch_size = driver.env.batch_size
        
#         dataset_it = iter(
#             replay_buffer.as_dataset(
#                 sample_batch_size=batch_size,
#                 num_steps=steps,
#                 single_deterministic_pass=True
#             )
#         )
#         log_interval = 1
        
#         for batch_id in range(async_steps_per_loop):
#             experience, unused_buffer_info = dataset_it.get_next()
#             set_expected_shape(experience, steps)
#             loss_info = agent.train(experience)
#             export_utils.export_metrics(
#                 step=train_step * async_steps_per_loop + batch_id,
#                 metrics=[],
#                 loss_info=loss_info
#             )
#         if log_interval and train_step % log_interval == 0:
#             print('step = {0}: loss = {1}'.format(train_step, round(loss_info.loss.numpy(), 2)))

#         replay_buffer.clear()

#     return training_loop