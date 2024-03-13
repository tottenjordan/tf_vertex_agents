"""The entrypoint for training a policy."""
import json
import os
import sys
import time
import random
import string
import argparse
import functools
from typing import List, Union
from pprint import pprint
import pickle as pkl
import numpy as np

from tensorflow.python.client import device_lib

# logging
import logging
# logging.disable(logging.WARNING)

from google.cloud import aiplatform as vertex_ai
from google.cloud import storage

# tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# TF-Agent agents & networks
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.agents import neural_linucb_agent
from tf_agents.bandits.agents import neural_epsilon_greedy_agent
from tf_agents.bandits.networks import global_and_arm_feature_network

# TF-Agent metrics
from tf_agents.bandits.metrics import tf_metrics as tf_bandit_metrics
from tf_agents.bandits.policies import policy_utilities

# TF-Agent drivers & buffers
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.train.utils import train_utils as tfa_train_utils

from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.trajectories import trajectory
from tf_agents.policies import py_tf_eager_policy
from tf_agents.train.utils import strategy_utils
from tf_agents.policies import policy_saver
from tf_agents.specs import array_spec
from tensorflow.python.client import device_lib

# this repo
from . import eval_perarm as eval_perarm
from . import train_perarm as train_perarm
from src.data import data_utils as data_utils
from src import train_utils as train_utils
from src import reward_factory as reward_factory
from src.agents import agent_factory as agent_factory
from src.networks import encoding_network as emb_features

if tf.__version__[0] != "2":
    raise Exception("The trainer only runs with TensorFlow version 2.")
    
import wrapt
print(f"wrapt version: {wrapt.__version__}")

PER_ARM = True  # Use the non-per-arm version of the MovieLens environment.

# clients
project_number = os.environ["CLOUD_ML_PROJECT_ID"]
storage_client = storage.Client(project=project_number)
# ====================================================
# get train & val datasets
# ====================================================
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO

# ====================================================
# Args
# ====================================================
def get_args(raw_args: List[str]) -> argparse.Namespace:
    """Parses parameters and hyperparameters for training a policy.

    Args:
      raw_args: A list of command line arguments.

    Re  turns:NETWORK_TYPE
    An argpase.Namespace object mapping (hyper)parameter names to the parsed
        values.
  """
    parser = argparse.ArgumentParser()

    # Path parameters
    parser.add_argument("--project_number", default="934903580331", type=str)
    parser.add_argument("--project", default="hybrid-vertex", type=str)
    parser.add_argument("--bucket_name", default="tmp", type=str)
    parser.add_argument("--artifacts_dir", type=str)
    parser.add_argument("--chkpoint_dir", default=None, type=str, help="Dir for storing checkpoints")
    parser.add_argument("--log_dir", default=None, type=str, help="Dir for TB logs")
    parser.add_argument("--data_dir_prefix_path", type=str, help="gs://{bucket_name}/{data_dir_prefix_path}/..")
    parser.add_argument("--vocab_prefix_path", default="vocabs", type=str)
    parser.add_argument("--vocab_filename", default="vocab_dict.pkl", type=str)
    # job config
    parser.add_argument("--distribute", default="single", type=str, help="")
    parser.add_argument("--experiment_name", default="tmp-experiment", type=str)
    parser.add_argument("--experiment_run", default="tmp-experiment-run", type=str)
    parser.add_argument("--agent_type", default="epsGreedy", type=str, help="'LinUCB' | 'LinTS |, 'epsGreedy' | 'NeuralLinUCB'")
    parser.add_argument("--network_type", default="dotproduct", type=str, help="'commontower' | 'dotproduct'")
    # Hyperparameters
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--eval_batch_size", default=1, type=int)
    parser.add_argument("--training_loops", default=4, type=int, help="Number of training iterations.")
    parser.add_argument("--num_epochs", default=4, type=int, help="Number of cycle through train data.")
    parser.add_argument("--steps_per_loop", default=2, type=int)
    parser.add_argument("--num_eval_steps", default=1000, type=int)
    parser.add_argument("--rank_k", default=20, type=int)
    parser.add_argument("--num_actions", default=20, type=int, help="Number of actions (movie items) to choose from.")
    # agent & network config
    parser.add_argument("--async_steps_per_loop", type=int, default=1, help="")
    parser.add_argument("--global_dim", type=int, default=1, help="")
    parser.add_argument("--per_arm_dim", type=int, default=1, help="")
    parser.add_argument("--resume_training_loops", action='store_true', help="include for True; ommit for False")
    parser.add_argument("--split", default=None, type=str, help="data split")
    parser.add_argument("--log_interval", type=int, default=1, help="")
    parser.add_argument("--chkpt_interval", type=int, default=100, help="")
    parser.add_argument('--global_layers', type=str, required=False)
    parser.add_argument('--arm_layers', type=str, required=False)
    parser.add_argument('--common_layers', type=str, required=False)
    parser.add_argument("--num_oov_buckets", type=int, default=1, help="")
    parser.add_argument("--global_emb_size", type=int, default=1, help="")
    parser.add_argument("--mv_emb_size", type=int, default=1, help="")
    parser.add_argument("--agent_alpha", type=float, default=0.1, help="")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="")
    parser.add_argument("--epsilon", type=float, default=0.01, help="")
    parser.add_argument("--encoding_dim", type=int, default=1, help="")
    parser.add_argument("--eps_phase_steps", type=int, default=1000, help="")
    # performance
    parser.add_argument('--tf_gpu_thread_count', type=str, required=False)
    parser.add_argument("--use_gpu", action='store_true', help="include for True; ommit for False")
    parser.add_argument("--use_tpu", action='store_true', help="include for True; ommit for False")
    parser.add_argument("--profiler", action='store_true', help="include for True; ommit for False")
    parser.add_argument("--sum_grads_vars", action='store_true', help="include for True; ommit for False")
    parser.add_argument("--debug_summaries", action='store_true', help="include for True; ommit for False")
    parser.add_argument("--cache_train", action='store_true', help="include for True; ommit for False")
    parser.add_argument("--is_testing", action='store_true', help="include for True; ommit for False")

    return parser.parse_args(raw_args)

def main(args: argparse.Namespace):
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
    print("logging args....")
    print(args)
    
    # =============================================
    # limiting GPU growth
    # =============================================
    if args.use_gpu:
        print("limiting GPU growth....")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f'detected: {len(gpus)} GPUs')
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        # tf.debugging.set_log_device_placement(True) # logs all tf ops and their device placement;
        os.environ['TF_GPU_THREAD_MODE']='gpu_private'
        os.environ['TF_GPU_THREAD_COUNT']= f'{args.tf_gpu_thread_count}'
        os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
    
    # =============================================
    # set GCP clients
    # =============================================
    from google.cloud import aiplatform as vertex_ai
    from google.cloud import storage

    project_number = os.environ["CLOUD_ML_PROJECT_ID"]
    storage_client = storage.Client(project=project_number)
    
    vertex_ai.init(
        project=project_number
        , location='us-central1'
        , experiment=args.experiment_name
    )
    # ====================================================
    # Set env variables
    # ====================================================

    GLOBAL_LAYERS = train_utils.get_arch_from_string(args.global_layers)
    ARM_LAYERS = train_utils.get_arch_from_string(args.arm_layers)
    COMMON_LAYERS = train_utils.get_arch_from_string(args.common_layers)
    
    print(f'GLOBAL_LAYERS = {GLOBAL_LAYERS}')
    print(f'ARM_LAYERS    = {ARM_LAYERS}')
    print(f'COMMON_LAYERS = {COMMON_LAYERS}')
    
    TOTAL_TRAIN_TAKE = args.training_loops * args.batch_size
    print(f'TOTAL_TRAIN_TAKE = {TOTAL_TRAIN_TAKE}')
    # ====================================================
    # Set Device Strategy
    # ====================================================
    print("Detecting devices....")
    print("Setting device strategy...")
    
    # GPU - All variables and Agents need to be created under strategy.scope()
    if args.use_gpu:
        distribution_strategy = strategy_utils.get_strategy(tpu=args.use_tpu, use_gpu=args.use_gpu)
        # distribution_strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        
    if args.use_tpu:
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        distribution_strategy = tf.distribute.TPUStrategy(cluster_resolver)
        logging.info("All devices: ", tf.config.list_logical_devices('TPU'))
        
    print(f"distribution_strategy: {distribution_strategy}")
    
    if distribution_strategy == 'multiworker':
        task_type, task_id = (
            strategy.cluster_resolver.task_type,
            strategy.cluster_resolver.task_id
        )
    else:
        task_type, task_id = 'chief', None
        
    NUM_REPLICAS = distribution_strategy.num_replicas_in_sync
    tf.print(f'NUM_REPLICAS = {NUM_REPLICAS}')
    tf.print(f'task_type = {task_type}')
    tf.print(f'task_id = {task_id}')
    # ====================================================
    # set Vertex AI env vars
    # ====================================================
    if 'AIP_TENSORBOARD_LOG_DIR' in os.environ:
        log_dir=os.environ['AIP_TENSORBOARD_LOG_DIR']
        tf.print(f'AIP_TENSORBOARD_LOG_DIR: {log_dir}')
    else:
        log_dir = args.log_dir
        tf.print(f'log_dir: {log_dir}')
        
    tf.print(f'TensorBoard log_dir: {log_dir}')
    
    # [Do Not Change] Set the root directory for training artifacts.
    MODEL_DIR = os.environ["AIP_MODEL_DIR"]
    tf.print(f'MODEL_DIR: {MODEL_DIR}')
    
    # ====================================================
    # Vocab Files
    # ====================================================
    EXISTING_VOCAB_FILE = f'gs://{args.bucket_name}/{args.vocab_prefix_path}/{args.vocab_filename}'
    tf.print(f'Downloading vocab file from: {EXISTING_VOCAB_FILE}...')
    
    data_utils.download_blob(
        project_id = args.project,
        bucket_name = args.bucket_name, 
        source_blob_name = f"{args.vocab_prefix_path}/{args.vocab_filename}", 
        destination_file_name= args.vocab_filename
    )

    tf.print(f"Downloaded vocab from: {EXISTING_VOCAB_FILE}\n")

    filehandler = open(f"{args.vocab_filename}", 'rb')
    VOCAB_DICT = pkl.load(filehandler)
    filehandler.close()
    
    # ====================================================
    # trajectory_fn
    # ====================================================
    with distribution_strategy.scope():
        
        embs = emb_features.EmbeddingModel(
            vocab_dict = VOCAB_DICT,
            num_oov_buckets = args.num_oov_buckets,
            global_emb_size = args.global_emb_size,
            mv_emb_size = args.mv_emb_size,
        )

        def _trajectory_fn(element):

            """
            Converts a dataset element into a trajectory.
            """
            global_features = embs._get_global_context_features(element)
            arm_features = embs._get_per_arm_features(element)

            # Adds a time dimension.
            arm_features = train_utils._add_outer_dimension(arm_features)

            # obs spec
            observation = {
                bandit_spec_utils.GLOBAL_FEATURE_KEY:
                    train_utils._add_outer_dimension(global_features)
            }

            reward = train_utils._add_outer_dimension(reward_factory._get_rewards(element))

            # To emit the predicted rewards in policy_info, we need to create dummy
            # rewards to match the definition in TensorSpec for the ones specified in
            # emit_policy_info set.
            dummy_rewards = tf.zeros([args.batch_size, 1, args.num_actions])
            policy_info = policy_utilities.PerArmPolicyInfo(
                chosen_arm_features=arm_features,
                # Pass dummy mean rewards here to match the model_spec for emitting
                # mean rewards in policy info
                predicted_rewards_mean=dummy_rewards,
                bandit_policy_type=tf.zeros([args.batch_size, 1, 1], dtype=tf.int32)
            )

            if args.agent_type == 'neural_ucb':
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
                discount=tf.zeros_like(reward)
            )
    
    # ====================================================
    # create agent
    # ====================================================
    observation_spec = {
        'global': tf.TensorSpec([args.global_dim], tf.float32),
        'per_arm': tf.TensorSpec([args.num_actions, args.per_arm_dim], tf.float32)
    }
    action_spec = tensor_spec.BoundedTensorSpec(
        shape=[], 
        dtype=tf.int32,
        minimum=tf.constant(0),            
        maximum=args.num_actions-1, 
        # n degrees of freedom and will dictate 
        # the expected mean reward spec shape
        name="action_spec",
    )
    time_step_spec = ts.time_step_spec(
        observation_spec = observation_spec, 
    )
    reward_spec = {
        "reward": array_spec.ArraySpec(
            shape=[args.batch_size], 
            dtype=np.float32, 
            name="reward"
        )
    }
    reward_tensor_spec = train_utils.from_spec(reward_spec)

    tf.print(f"observation_spec : {observation_spec}")
    tf.print(f"action_spec      : {action_spec}")
    tf.print(f"time_step_spec   : {time_step_spec}")

    with distribution_strategy.scope():

        global_step = tf.compat.v1.train.get_or_create_global_step()
    
        agent = agent_factory.PerArmAgentFactory._get_agent(
            agent_type=args.agent_type, 
            network_type=args.network_type, 
            time_step_spec=time_step_spec, 
            action_spec=action_spec, 
            observation_spec=observation_spec,
            global_layers = GLOBAL_LAYERS,
            arm_layers = ARM_LAYERS,
            common_layers = COMMON_LAYERS,
            agent_alpha = args.agent_alpha,
            learning_rate = args.learning_rate,
            epsilon = args.epsilon,
            train_step_counter = global_step,
            output_dim = args.encoding_dim,
            eps_phase_steps = args.eps_phase_steps,
            summarize_grads_and_vars = args.sum_grads_vars,
            debug_summaries = args.debug_summaries
        )
    
        agent.initialize()
    tf.print(f"agent: {agent.name}")
    tf.print(f"network_type: {args.network_type}")
    tf.print(f"global_step: {global_step.value().numpy()}")
    
    tf.print("Inpsecting agent policy from task file...")
    tf.print(f"agent.policy: {agent.policy}")
    tf.print("Inpsecting agent policy from task file: Complete")

    # ====================================================
    # val dataset
    # ====================================================
    val_dataset = train_utils._get_eval_dataset(
        args.bucket_name, 
        args.data_dir_prefix_path, 
        split="val", 
        batch_size=args.eval_batch_size
    )
    eval_ds = val_dataset.batch(args.eval_batch_size)

    if args.num_eval_steps > 0:
        eval_ds = eval_ds.take(args.num_eval_steps)

    with distribution_strategy.scope():
        eval_ds = eval_ds.cache()

    tf.print(f"eval_ds: {eval_ds}")
    # ====================================================
    # TB summary writer
    # ====================================================
    tf.print(f"log_dir: {log_dir}")
    tf.print(f"current thread has eager execution enabled: {tf.executing_eagerly()}")

    with distribution_strategy.scope():
        train_summary_writer = tf.compat.v2.summary.create_file_writer(
            log_dir, flush_millis=10 * 1000
        )
        train_summary_writer.set_as_default()

    # ====================================================
    # train loop
    # ====================================================
    # Reset the train step
    # agent.train_step_counter.assign(0)

    # start the timer and training
    start_time = time.time()

    metric_results, trained_agent = train_perarm.train_perarm(
        agent = agent,
        epsilon = args.epsilon,
        reward_spec = reward_tensor_spec,
        global_dim = args.global_dim,
        per_arm_dim = args.per_arm_dim,
        num_iterations = args.training_loops,
        steps_per_loop = args.steps_per_loop,
        num_eval_steps = args.num_eval_steps,
        # data
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        # functions
        _trajectory_fn = _trajectory_fn,
        # _run_bandit_eval_fn = _run_bandit_eval,
        # train intervals
        chkpt_interval = args.chkpt_interval,
        log_interval = args.log_interval,
        # dirs
        bucket_name=args.bucket_name,
        data_dir_prefix_path=args.data_dir_prefix_path,
        log_dir=args.log_dir,
        model_dir=args.artifacts_dir,
        # root_dir=args.root_dir,
        chkpoint_dir=args.chkpoint_dir,
        async_steps_per_loop = args.async_steps_per_loop,
        resume_training_loops = args.resume_training_loops,
        use_gpu=args.use_gpu,
        use_tpu=args.use_tpu,
        profiler=args.profiler,
        train_summary_writer = train_summary_writer,
        total_train_take = TOTAL_TRAIN_TAKE,
        global_step = global_step,
        num_replicas = NUM_REPLICAS,
        cache_train_data = args.cache_train,
        strategy = distribution_strategy,
        # saver = saver,
        is_testing=args.is_testing,
        num_epochs=args.num_epochs,
    )

    end_time = time.time()
    runtime_mins = int((end_time - start_time) / 60)
    tf.print(f"complete train job in {runtime_mins} minutes")
    tf.print(f"trained_agent: {trained_agent}")
    
    # ====================================================
    # Evaluate the agent's policy once after training
    # ====================================================
    tf.print(f"Load trained policy & evaluate...")
    tf.print(f"load policy from model_dir: {args.artifacts_dir}")
 
    trained_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        args.artifacts_dir, 
        load_specs_from_pbtxt=True
    )
    tf.print(f"trained_policy: {trained_policy}")
    tf.print(f"evaluating trained Policy...")
    start_time = time.time()

    val_loss, preds, tr_rewards = eval_perarm._run_bandit_eval(
        policy = trained_policy,
        data = eval_ds, # eval_ds | dist_eval_ds
        eval_batch_size = args.eval_batch_size,
        per_arm_dim = args.per_arm_dim,
        global_dim = args.global_dim,
        vocab_dict = VOCAB_DICT,
        num_oov_buckets = args.num_oov_buckets,
        global_emb_size = args.global_emb_size,
        mv_emb_size = args.mv_emb_size,
    )

    runtime_mins = int((time.time() - start_time) / 60)
    tf.print(f"post-train val_loss: {val_loss}")
    tf.print(f"post-train eval runtime : {runtime_mins}")
    
    # ====================================================
    # log Vertex Experiments
    # ====================================================
    if task_type == 'chief':
        tf.print(f" task_type logging experiments: {task_type}")
        tf.print(f" task_id logging experiments: {task_id}")
        tf.print(f" logging data to experiment run: {args.experiment_run}")

        with vertex_ai.start_run(
            f'{args.experiment_run}',
            # tensorboard=args.tb_resource_name
        ) as my_run:

            # tf.print(f"logging time-series metrics...")
            # for i in range(len(metric_results)):
            #     vertex_ai.log_time_series_metrics({'loss': metric_results[i]}, step=i)

            tf.print(f"logging metrics...")
            # gather the metrics for the last epoch to be saved in metrics
            my_run.log_metrics(
                {
                    "train_loss" : round(float(metric_results[-1]),2)
                    , "val_loss" : round(float(val_loss.numpy()),2)
                }
            )

            tf.print(f"logging metaparams...")
            my_run.log_params(
                {
                    "agent_type": agent.name,
                    "network": args.network_type,
                    "runtime": runtime_mins,
                    "batch_size": args.batch_size, 
                    "training_loops": args.training_loops,
                    "global_lyrs": args.global_layers,
                    "arm_lyrs": args.arm_layers,
                    "common_lyrs": args.common_layers,
                    "encoding_dim": args.encoding_dim,
                }
            )

            vertex_ai.end_run()
            tf.print(f"EXPERIMENT RUN: {args.experiment_run} has ended")

if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        level=logging.INFO, 
        datefmt='%d-%m-%y %H:%M:%S',
        stream=sys.stdout
    )
    logging.info("Python Version = %s", sys.version)
    logging.info("TensorFlow Version = %s", tf.__version__)
    logging.info("Reinforcement learning task started...")

    args = get_args(sys.argv[1:])
    logging.info('Args: %s', args)

    main(args = args)

    logging.info("Reinforcement learning task completed.")