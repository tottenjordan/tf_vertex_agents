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

"""The entrypoint for training a policy."""
import argparse
import functools
import json
import logging
import os
import sys
from typing import List, Union

# google cloud
from google.cloud import aiplatform, storage
import hypertune

from . import policy_util
from . import data_utils
from . import train_utils
from . import data_config
from . import my_per_arm_py_env

import tensorflow as tf
from tensorflow.python.client import device_lib
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.environments import environment_utilities
from tf_agents.bandits.environments import movielens_py_environment
from tf_agents.bandits.metrics import tf_metrics as tf_bandit_metrics
from tf_agents.environments import tf_py_environment

if tf.__version__[0] != "2":
    raise Exception("The trainer only runs with TensorFlow version 2.")

PER_ARM = True  # Use the non-per-arm version of the MovieLens environment.

def get_args(
    raw_args: List[str]
) -> argparse.Namespace:
    """Parses parameters and hyperparameters for training a policy.

    Args:
      raw_args: A list of command line arguments.

    Returns:
      An argpase.Namespace object mapping (hyper)parameter names to the parsed
      values.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--project_id"
        , type=str
        , default='hybrid-vertex'
    )
    # Whether to execute hyperparameter tuning or training
    parser.add_argument(
        "--run-hyperparameter-tuning"
        , action="store_true"
        , help="Whether to perform hyperparameter tuning instead of regular training."
    )
    # Whether to train using the best hyperparameters learned from a previous
    # hyperparameter tuning job.
    parser.add_argument(
        "--train-with-best-hyperparameters"
        , action="store_true"
        , help="Whether to train using the best hyperparameters learned from a previous hyperparameter tuning job."
    )
    # Path parameters
    parser.add_argument(
        "--artifacts-dir"
        , type=str
        , help="Extra directory where model artifacts are saved."
    )
    parser.add_argument(
        "--profiler-dir"
        , default=None
        , type=str
        , help="Directory for TensorBoard Profiler artifacts."
    )
    parser.add_argument(
        "--data-path", type=str, help="Path to MovieLens 100K's 'u.data' file."
    )
    parser.add_argument(
        "--best-hyperparameters-bucket"
        , type=str
        , help="Path to MovieLens 100K's 'u.data' file."
    )
    parser.add_argument(
        "--best-hyperparameters-path"
        , type=str
        , help="Path to JSON file containing the best hyperparameters."
    )
    # Hyperparameters
    parser.add_argument(
        "--batch-size"
        , default=8
        , type=int
        , help="Training and prediction batch size."
    )
    parser.add_argument(
        "--training-loops"
        , default=4
        , type=int
        , help="Number of training iterations."
    )
    parser.add_argument(
        "--steps-per-loop"
        , default=2
        , type=int
        , help="Number of driver steps per training iteration."
    )
    # MovieLens simulation environment parameters
    parser.add_argument(
        "--rank-k"
        , default=20
        , type=int
        , help="Rank for matrix factorization in the MovieLens environment; also the observation dimension."
    )
    parser.add_argument(
        "--num-actions"
        , default=20
        , type=int
        , help="Number of actions (movie items) to choose from."
    )
    # LinUCB agent parameters
    parser.add_argument(
        "--tikhonov-weight"
        , default=0.001
        , type=float
        , help="LinUCB Tikhonov regularization weight."
    )
    parser.add_argument(
        "--agent-alpha"
        , default=10.0
        , type=float
        , help="LinUCB exploration parameter that multiplies the confidence intervals."
    )

    ### new
    parser.add_argument(
        "--bucket_name"
        , default="tmp"
        , type=str
        , help=" "
    )

    parser.add_argument(
        "--data_gcs_prefix"
        , default="data"
        , type=str
        , help=""
    )
    
    parser.add_argument(
        "--data_path"
        , default="gs://tmp/tmp"
        , type=str
        , help=""
    )
    
    parser.add_argument(
        "--project_number"
        , default="934903580331"
        , type=str
        , help=""
    )
    # distribute
    parser.add_argument(
        "--distribute"
        , default="single"
        , type=str
        , help=""
    )
    # artifacts_dir
    parser.add_argument(
        "--artifacts_dir"
        , default="gs://BUCKET/EXPERIMENT/RUN_NAME/artifacts"
        , type=str
        , help=""
    )
    parser.add_argument(
        "--root_dir"
        , default="gs://BUCKET/EXPERIMENT/RUN_NAME/root"
        , type=str
        , help=""
    )
    
    return parser.parse_args(raw_args)

def execute_task(
    args: argparse.Namespace
    , best_hyperparameters_blob: Union[storage.Blob, None]
    , hypertune_client: Union[hypertune.HyperTune, None]
) -> None:
    """Executes training, or hyperparameter tuning, for the policy.

    Parses parameters and hyperparameters from the command line, reads best
    hyperparameters if applicable, constructs the logical modules for RL, and
    executes training or hyperparameter tuning. Tracks the training process
    and resources using TensorBoard Profiler if applicable.

    Args:
      args: An argpase.Namespace object of (hyper)parameter values.
      best_hyperparameters_blob: An object containing best hyperparameters in
        Google Cloud Storage.
      hypertune_client: Client for submitting hyperparameter tuning metrics.
    """
    
    # [Do Not Change] Set the root directory for training artifacts.
    # TODO - JT
    MODEL_DIR = os.environ["AIP_MODEL_DIR"] if not args.run_hyperparameter_tuning else ""
    root_dir = args.root_dir if not args.run_hyperparameter_tuning else ""
    logging.info(f'root_dir: {root_dir}')

    # Use best hyperparameters learned from a previous hyperparameter tuning job.
    logging.info(args.train_with_best_hyperparameters)
    if args.train_with_best_hyperparameters:
        logging.info(f'train_with_best_hyperparameters engaged...')
        best_hyperparameters = json.loads(
            best_hyperparameters_blob.download_as_string()
        )
        
        if "batch-size" in best_hyperparameters:
            args.batch_size = int(best_hyperparameters["batch-size"])
        if "training-loops" in best_hyperparameters:
            args.training_loops = int(best_hyperparameters["training-loops"])
        if "steps-per-loop" in best_hyperparameters:
            args.step_per_loop = int(best_hyperparameters["steps-per-loop"])

    # Define RL environment.
    env = my_per_arm_py_env.MyMovieLensPerArmPyEnvironment(
        project_number = args.project_number
        , data_path = args.data_path
        , bucket_name = args.bucket_name
        , data_gcs_prefix = args.data_gcs_prefix
        , user_age_lookup_dict = data_config.USER_AGE_LOOKUP
        , user_occ_lookup_dict = data_config.USER_OCC_LOOKUP
        , movie_gen_lookup_dict = data_config.MOVIE_GEN_LOOKUP
        , num_users = data_config.MOVIELENS_NUM_USERS
        , num_movies = data_config.MOVIELENS_NUM_MOVIES
        , rank_k = args.rank_k
        , batch_size = args.batch_size
        , num_actions = args.num_actions
    )
    environment = tf_py_environment.TFPyEnvironment(env)
    
    strategy = train_utils.get_train_strategy(distribute_arg=args.distribute)
    logging.info(f'TF training strategy (execute task) = {strategy}')
    
    with strategy.scope():
        # Define RL agent/algorithm.
        agent = lin_ucb_agent.LinearUCBAgent(
            time_step_spec=environment.time_step_spec()
            , action_spec=environment.action_spec()
            , tikhonov_weight=args.tikhonov_weight
            , alpha=args.agent_alpha
            , dtype=tf.float32
            , accepts_per_arm_features=PER_ARM # TODO - streamline
        )
    logging.info("TimeStep Spec (for each batch):\n%s\n", agent.time_step_spec)
    logging.info("Action Spec (for each batch):\n%s\n", agent.action_spec)
    logging.info("Reward Spec (for each batch):\n%s\n", environment.reward_spec())

    # Define RL metric.
    optimal_reward_fn = functools.partial(
        environment_utilities.compute_optimal_reward_with_movielens_environment
        , environment=environment
    )
    
    regret_metric = tf_bandit_metrics.RegretMetric(optimal_reward_fn)
    metrics = [regret_metric]

    # Perform on-policy training with the simulation MovieLens environment.
    if args.profiler_dir is not None:
        tf.profiler.experimental.start(args.profiler_dir)
  
    metric_results = policy_util.train(
        agent=agent
        , environment=environment
        , training_loops=args.training_loops
        , steps_per_loop=args.steps_per_loop
        , additional_metrics=metrics
        , run_hyperparameter_tuning=args.run_hyperparameter_tuning
        , root_dir=root_dir if not args.run_hyperparameter_tuning else None
        , artifacts_dir=args.artifacts_dir
        if not args.run_hyperparameter_tuning else None
        , model_dir = MODEL_DIR
    )
    
    if args.profiler_dir is not None:
        tf.profiler.experimental.stop()

    # Report training metrics to Vertex AI for hyperparameter tuning
    if args.run_hyperparameter_tuning:
        hypertune_client.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag="final_average_return"
            , metric_value=metric_results["AverageReturnMetric"][-1]
            # , global_step=args.training_loops
        )

def main() -> None:
    """
    Entry point for training or hyperparameter tuning.
    """
    args = get_args(sys.argv[1:])
    # =============================================
    # set GCP clients
    # =============================================
    from google.cloud import aiplatform as vertex_ai
    from google.cloud import storage

    project_number = os.environ["CLOUD_ML_PROJECT_ID"]
    storage_client = storage.Client(project=project_number)
    
    vertex_ai.init(
        project=project_number,
        location='us-central1',
        # experiment=args.experiment_name
    )
    
    # =============================================
    # GPUs
    # =============================================

    # limiting GPU growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f'detected: {len(gpus)} GPUs')
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logging.info(e)

    # tf.debugging.set_log_device_placement(True)          # logs all tf ops and their device placement;
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = f'8'               # TODO - parametrize | 1
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    
    # ====================================================
    # Set Device Strategy
    # ====================================================
    logging.info("Detecting devices....")
    logging.info('DEVICES'  + str(device_lib.list_local_devices()))
    
    logging.info("Setting device strategy...")
    
    strategy = train_utils.get_train_strategy(distribute_arg=args.distribute)
    logging.info(f'TF training strategy (main) = {strategy}')
    
    NUM_REPLICAS = strategy.num_replicas_in_sync
    logging.info(f'num_replicas_in_sync = {NUM_REPLICAS}')
    
    # Here the batch size scales up by number of workers since
    # `tf.data.Dataset.batch` expects the global batch size.
    GLOBAL_BATCH_SIZE = int(args.batch_size) * int(NUM_REPLICAS)
    logging.info(f'GLOBAL_BATCH_SIZE = {GLOBAL_BATCH_SIZE}')

    # type and task of machine from strategy
    logging.info(f'Setting task_type and task_id...')
    if args.distribute == 'multiworker':
        task_type, task_id = (
            strategy.cluster_resolver.task_type,
            strategy.cluster_resolver.task_id
        )
    else:
        task_type, task_id = 'chief', None
    
    logging.info(f'task_type = {task_type}')
    logging.info(f'task_id = {task_id}')
    
    # ====================================================
    # determine train job type and execute
    # ====================================================
    
    if args.train_with_best_hyperparameters:
        storage_client = storage.Client(args.project_id)
        bucket = storage_client.bucket(args.bucket_name)
        best_hyperparameters_blob = bucket.blob(args.best_hyperparameters_path)
    
    else:
        best_hyperparameters_blob = None
    
    hypertune_client = hypertune.HyperTune() if args.run_hyperparameter_tuning else None

    execute_task(
        args = args
        , best_hyperparameters_blob = best_hyperparameters_blob
        , hypertune_client = hypertune_client
    )

if __name__ == "__main__":
    
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Python Version = %s", sys.version)
    logging.info("TensorFlow Version = %s", tf.__version__)
    # logging.info("TF_CONFIG = %s", os.environ.get("TF_CONFIG", "Not found"))
    # logging.info("DEVICES = %s", device_lib.list_local_devices())
    logging.info("Reinforcement learning task started...")
    
    main()
    
    logging.info("Reinforcement learning task completed.")
