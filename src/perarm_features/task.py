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

# logging
import logging
logging.disable(logging.WARNING)

from google.cloud import aiplatform as vertex_ai
from google.cloud import storage

# tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# TF-Agent env
from tf_agents.bandits.environments import environment_utilities
from tf_agents.bandits.environments import stationary_stochastic_per_arm_py_environment as p_a_env
from tf_agents.environments import tf_py_environment

# TF-Agent agents & networks
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.agents import neural_linucb_agent
from tf_agents.bandits.agents import neural_epsilon_greedy_agent
from tf_agents.bandits.networks import global_and_arm_feature_network
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

nest = tf.nest

from . import trainer_common as trainer_common
from src.per_arm_rl import data_utils
from src.per_arm_rl import train_utils
from src.per_arm_rl import data_config

if tf.__version__[0] != "2":
    raise Exception("The trainer only runs with TensorFlow version 2.")


PER_ARM = True  # Use the non-per-arm version of the MovieLens environment.

# clients
project_number = os.environ["CLOUD_ML_PROJECT_ID"]
storage_client = storage.Client(project=project_number)
# vertex_ai.init(
#     project=project_number,
#     location='us-central1',
#     experiment=args.experiment_name
# )

# ====================================================
# get train & val datasets
# ====================================================
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO

def _get_train_dataset(bucket_name, data_dir_prefix_path, split):
    train_files = []
    for blob in storage_client.list_blobs(f"{bucket_name}", prefix=f'{data_dir_prefix_path}/{split}'): # tmp TODO - "train"
        if '.tfrecord' in blob.name:
            train_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))
            
    logging.info(f"train_files: {train_files}")

    train_dataset = tf.data.TFRecordDataset(train_files)
    train_dataset = train_dataset.map(data_utils.parse_tfrecord)
    
    return train_dataset

# ====================================================
# get global context (user) feature embedding models 
# -- TODO: parameterize
# ====================================================

def get_user_id_emb_model(vocab_dict):
    
    user_id_input_layer = tf.keras.Input(
        name="user_id",
        shape=(1,),
        dtype=tf.string
    )

    user_id_lookup = tf.keras.layers.StringLookup(
        max_tokens=len(vocab_dict['user_id']) + data_config.NUM_OOV_BUCKETS,
        num_oov_indices=data_config.NUM_OOV_BUCKETS,
        mask_token=None,
        vocabulary=vocab_dict['user_id'],
    )(user_id_input_layer)

    user_id_embedding = tf.keras.layers.Embedding(
        # Let's use the explicit vocabulary lookup.
        input_dim=len(vocab_dict['user_id']) + data_config.NUM_OOV_BUCKETS,
        output_dim=data_config.GLOBAL_EMBEDDING_SIZE
    )(user_id_lookup)
    
    user_id_embedding = tf.reduce_sum(user_id_embedding, axis=-2)
    user_id_model = tf.keras.Model(inputs=user_id_input_layer, outputs=user_id_embedding)
    
    return user_id_model

def get_user_age_emb_model(vocab_dict):
    
    user_age_input_layer = tf.keras.Input(
        name="bucketized_user_age",
        shape=(1,),
        dtype=tf.float32
    )

    user_age_lookup = tf.keras.layers.IntegerLookup(
        vocabulary=vocab_dict['bucketized_user_age'],
        num_oov_indices=data_config.NUM_OOV_BUCKETS,
        oov_value=0,
    )(user_age_input_layer)

    user_age_embedding = tf.keras.layers.Embedding(
        # Let's use the explicit vocabulary lookup.
        input_dim=len(vocab_dict['bucketized_user_age']) + data_config.NUM_OOV_BUCKETS,
        output_dim=data_config.GLOBAL_EMBEDDING_SIZE
    )(user_age_lookup)

    user_age_embedding = tf.reduce_sum(user_age_embedding, axis=-2)
    user_age_model = tf.keras.Model(inputs=user_age_input_layer, outputs=user_age_embedding)
    
    return user_age_model

def get_user_occ_emb_model(vocab_dict):
    
    user_occ_input_layer = tf.keras.Input(
        name="user_occupation_text",
        shape=(1,),
        dtype=tf.string
    )
    user_occ_lookup = tf.keras.layers.StringLookup(
        max_tokens=len(vocab_dict['user_occupation_text']) + data_config.NUM_OOV_BUCKETS,
        num_oov_indices=data_config.NUM_OOV_BUCKETS,
        mask_token=None,
        vocabulary=vocab_dict['user_occupation_text'],
    )(user_occ_input_layer)
    
    user_occ_embedding = tf.keras.layers.Embedding(
        # Let's use the explicit vocabulary lookup.
        input_dim=len(vocab_dict['user_occupation_text']) + data_config.NUM_OOV_BUCKETS,
        output_dim=data_config.GLOBAL_EMBEDDING_SIZE
    )(user_occ_lookup)
    
    user_occ_embedding = tf.reduce_sum(user_occ_embedding, axis=-2)
    user_occ_model = tf.keras.Model(inputs=user_occ_input_layer, outputs=user_occ_embedding)
    
    return user_occ_model

def get_ts_emb_model(vocab_dict):
    
    user_ts_input_layer = tf.keras.Input(
        name="timestamp",
        shape=(1,),
        dtype=tf.int64
    )

    user_ts_lookup = tf.keras.layers.Discretization(
        vocab_dict['timestamp_buckets'].tolist()
    )(user_ts_input_layer)

    user_ts_embedding = tf.keras.layers.Embedding(
        # Let's use the explicit vocabulary lookup.
        input_dim=len(vocab_dict['timestamp_buckets'].tolist()) + data_config.NUM_OOV_BUCKETS,
        output_dim=data_config.GLOBAL_EMBEDDING_SIZE
    )(user_ts_lookup)

    user_ts_embedding = tf.reduce_sum(user_ts_embedding, axis=-2)
    user_ts_model = tf.keras.Model(inputs=user_ts_input_layer, outputs=user_ts_embedding)
    
    return user_ts_model

# ====================================================
# get perarm feature embedding models
# -- TODO: parameterize
# ====================================================

def get_mv_id_emb_model(vocab_dict):
    
    mv_id_input_layer = tf.keras.Input(
        name="movie_id",
        shape=(1,),
        dtype=tf.string
    )

    mv_id_lookup = tf.keras.layers.StringLookup(
        max_tokens=len(vocab_dict['movie_id']) + data_config.NUM_OOV_BUCKETS,
        num_oov_indices=data_config.NUM_OOV_BUCKETS,
        mask_token=None,
        vocabulary=vocab_dict['movie_id'],
    )(mv_id_input_layer)

    mv_id_embedding = tf.keras.layers.Embedding(
        # Let's use the explicit vocabulary lookup.
        input_dim=len(vocab_dict['movie_id']) + data_config.NUM_OOV_BUCKETS,
        output_dim=data_config.MV_EMBEDDING_SIZE
    )(mv_id_lookup)

    mv_id_embedding = tf.reduce_sum(mv_id_embedding, axis=-2)
    mv_id_model = tf.keras.Model(inputs=mv_id_input_layer, outputs=mv_id_embedding)
    
    return mv_id_model

def get_mv_gen_emb_model(vocab_dict):
    
    mv_genre_input_layer = tf.keras.Input(
        name="movie_genres",
        shape=(1,),
        dtype=tf.float32
    )

    mv_genre_lookup = tf.keras.layers.IntegerLookup(
        vocabulary=vocab_dict['movie_genres'],
        num_oov_indices=data_config.NUM_OOV_BUCKETS,
        oov_value=0,
    )(mv_genre_input_layer)

    mv_genre_embedding = tf.keras.layers.Embedding(
        # Let's use the explicit vocabulary lookup.
        input_dim=len(vocab_dict['movie_genres']) + data_config.NUM_OOV_BUCKETS,
        output_dim=data_config.MV_EMBEDDING_SIZE
    )(mv_genre_lookup)

    mv_genre_embedding = tf.reduce_sum(mv_genre_embedding, axis=-2)
    mv_gen_model = tf.keras.Model(inputs=mv_genre_input_layer, outputs=mv_genre_embedding)
    
    return mv_gen_model

# ====================================================
# get agent
# ====================================================
observation_and_action_constraint_splitter = None

def _get_agent(agent_type, network_type, environment):
    network = None

    if agent_type == 'LinUCB':
        agent = lin_ucb_agent.LinearUCBAgent(
            time_step_spec=environment.time_step_spec(),
            action_spec=environment.action_spec(),
            alpha=data_config.AGENT_ALPHA,
            accepts_per_arm_features=PER_ARM,
            dtype=tf.float32,
        )
    elif agent_type == 'LinTS':
        agent = lin_ts_agent.LinearThompsonSamplingAgent(
            time_step_spec=environment.time_step_spec(),
            action_spec=environment.action_spec(),
            alpha=data_config.AGENT_ALPHA,
            observation_and_action_constraint_splitter=(
                observation_and_action_constraint_splitter
            ),
            accepts_per_arm_features=PER_ARM,
            dtype=tf.float32,
        )
    elif agent_type == 'epsGreedy':
        obs_spec = environment.observation_spec()
        if network_type == 'commontower':
            network = global_and_arm_feature_network.create_feed_forward_common_tower_network(
                observation_spec = obs_spec, 
                global_layers = data_config.GLOBAL_LAYERS, 
                arm_layers = data_config.ARM_LAYERS, 
                common_layers = data_config.COMMON_LAYERS,
                # output_dim = 1
            )
        elif network_type == 'dotproduct':
            network = global_and_arm_feature_network.create_feed_forward_dot_product_network(
                observation_spec = obs_spec, 
                global_layers = data_config.GLOBAL_LAYERS, 
                arm_layers = data_config.ARM_LAYERS
            )
        agent = neural_epsilon_greedy_agent.NeuralEpsilonGreedyAgent(
            time_step_spec=environment.time_step_spec(),
            action_spec=environment.action_spec(),
            reward_network=network,
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=data_config.LR),
            epsilon=data_config.EPSILON,
            observation_and_action_constraint_splitter=(
                observation_and_action_constraint_splitter
            ),
            accepts_per_arm_features=PER_ARM,
            emit_policy_info=policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN,
        )

    elif agent_type == 'NeuralLinUCB':
        obs_spec = environment.observation_spec()
        network = (
            global_and_arm_feature_network.create_feed_forward_common_tower_network(
                observation_spec = obs_spec, 
                global_layers = data_config.GLOBAL_LAYERS, 
                arm_layers = data_config.ARM_LAYERS, 
                common_layers = data_config.COMMON_LAYERS,
                output_dim = data_config.ENCODING_DIM
            )
        )
        agent = neural_linucb_agent.NeuralLinUCBAgent(
            time_step_spec=environment.time_step_spec(),
            action_spec=environment.action_spec(),
            encoding_network=network,
            encoding_network_num_train_steps=data_config.EPS_PHASE_STEPS,
            encoding_dim=data_config.ENCODING_DIM,
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=data_config.LR),
            alpha=1.0,
            gamma=1.0,
            epsilon_greedy=data_config.EPSILON,
            accepts_per_arm_features=PER_ARM,
            debug_summaries=True,
            summarize_grads_and_vars=True,
            emit_policy_info=policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN,
        )

    logging.info(f"Agent: {agent.name}\n")

    if network:
        logging.info(f"Network: {network.name}")
        network = network.name
    
    return agent, network

# ====================================================
# get rewards functions
# ====================================================
def _all_rewards(observation, hidden_param):
    """Outputs rewards for all actions, given an observation."""
    hidden_param = tf.cast(hidden_param, dtype=tf.float32)
    global_obs = observation['global']
    per_arm_obs = observation['per_arm']
    num_actions = tf.shape(per_arm_obs)[1]
    tiled_global = tf.tile(
        tf.expand_dims(global_obs, axis=1), [1, num_actions, 1])
    concatenated = tf.concat([tiled_global, per_arm_obs], axis=-1)
    rewards = tf.linalg.matvec(concatenated, hidden_param)
    return rewards

def optimal_reward(observation, hidden_param):
    """Outputs the maximum expected reward for every element in the batch."""
    return tf.reduce_max(
        _all_rewards(observation, hidden_param), axis=1
    )

def optimal_action(observation, hidden_param):
    return tf.argmax(
        _all_rewards(observation, hidden_param), axis=1, output_type=tf.int32
    )

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
    parser.add_argument("--root_dir", default=None, type=str, help="Dir for storing checkpoints")
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
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--training_loops", default=4, type=int, help="Number of training iterations.")
    parser.add_argument("--steps_per_loop", default=2, type=int)
    parser.add_argument("--rank_k", default=20, type=int)
    parser.add_argument("--num_actions", default=20, type=int, help="Number of actions (movie items) to choose from.")
    parser.add_argument("--reward_param", help="hidden_param (list)")
    parser.add_argument("--async_steps_per_loop", type=int, default=1, help="")

    return parser.parse_args(raw_args)

def execute_task(args: argparse.Namespace) -> None:
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
    
    # ====================================================
    # Set env variables
    # ====================================================
    REWARD_PARAM = train_utils.get_arch_from_string(args.reward_param)
    logging.info(f'REWARD_PARAM = {REWARD_PARAM}')
    
    # clients
    # storage_client = storage.Client(project=args.project)
    
    # vertex_ai.init(
    #     project=project_number,
    #     location='us-central1',
    #     experiment=args.experiment_name
    # )
    
    # ====================================================
    # Set Device Strategy
    # ====================================================
    logging.info("Detecting devices....")
    logging.info("Setting device strategy...")
    
    strategy = train_utils.get_train_strategy(distribute_arg=args.distribute)
    logging.info(f'TF training strategy (execute task) = {strategy}')
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
    # set Vertex AI env vars
    # ====================================================
    if 'AIP_TENSORBOARD_LOG_DIR' in os.environ:
        log_dir=os.environ['AIP_TENSORBOARD_LOG_DIR']
        logging.info(f'AIP_TENSORBOARD_LOG_DIR: {log_dir}')
    else:
        log_dir = args.log_dir
        logging.info(f'log_dir: {log_dir}')
        
    logging.info(f'TensorBoard log_dir: {log_dir}')
    
    # [Do Not Change] Set the root directory for training artifacts.
    MODEL_DIR = os.environ["AIP_MODEL_DIR"]
    logging.info(f'MODEL_DIR: {MODEL_DIR}')
    
    # root_dir = args.root_dir if not args.run_hyperparameter_tuning else ""
    # logging.info(f'root_dir: {root_dir}')
    
    # ====================================================
    # Vocab Files
    # ====================================================
    EXISTING_VOCAB_FILE = f'gs://{args.bucket_name}/{args.vocab_prefix_path}/{args.vocab_filename}'
    logging.info(f'Downloading vocab file from: {EXISTING_VOCAB_FILE}...')
    
    data_utils.download_blob(
        project_id = args.project,
        bucket_name = args.bucket_name, 
        source_blob_name = f"{args.vocab_prefix_path}/{args.vocab_filename}", 
        destination_file_name= args.vocab_filename
    )

    print(f"Downloaded vocab from: {EXISTING_VOCAB_FILE}\n")

    filehandler = open(f"{args.vocab_filename}", 'rb')
    VOCAB_DICT = pkl.load(filehandler)
    filehandler.close()
    
    # ====================================================
    # train dataset
    # ====================================================
    train_dataset = _get_train_dataset(args.bucket_name, args.data_dir_prefix_path, split="train")
    
    # ====================================================
    # get global_context_sampling_fn
    # ====================================================
    def global_context_sampling_fn():
        """
        This function generates a single global observation vector.
        """
        user_id_model = get_user_id_emb_model(VOCAB_DICT)
        user_age_model = get_user_age_emb_model(VOCAB_DICT)
        user_occ_model = get_user_occ_emb_model(VOCAB_DICT)
        user_ts_model = get_ts_emb_model(VOCAB_DICT)
        
        for x in train_dataset.batch(1).take(1):
            user_id_value = x['user_id']
            user_age_value = x['bucketized_user_age']
            user_occ_value = x['user_occupation_text']
            user_ts_value = x['timestamp']

            _id = user_id_model(user_id_value)
            _age = user_age_model(user_age_value)
            _occ = user_occ_model(user_occ_value)
            _ts = user_ts_model(user_ts_value)

            # to numpy array
            _id = np.array(_id.numpy()[0])
            _age = np.array(_age.numpy()[0])
            _occ = np.array(_occ.numpy()[0])
            _ts = np.array(_ts.numpy()[0])

            concat = np.concatenate(
                [_id, _age, _occ, _ts], axis=-1
            ).astype(np.float32)

            return concat
    
    # ====================================================
    # get per_arm_context_sampling_fn
    # ====================================================
    def per_arm_context_sampling_fn():
        """
        This function generates a single per-arm observation vector
        """

        mvid_model = get_mv_id_emb_model(VOCAB_DICT)
        mvgen_model = get_mv_gen_emb_model(VOCAB_DICT)
        
        for x in train_dataset.batch(1).take(1):
            mv_id_value = x['movie_id']
            mv_gen_value = x['movie_genres'][0]

            _mid = mvid_model(mv_id_value)
            _mgen = mvgen_model(mv_gen_value)

            # to numpy array
            _mid = np.array(_mid.numpy()[0])
            _mgen = np.array(_mgen.numpy()[0])

            concat = np.concatenate([_mid, _mgen], axis=-1).astype(np.float32)

            return concat
        
    # ====================================================
    # get reward_fn & action_fn
    # ====================================================
    class LinearNormalReward(object):
        
        def __init__(self, theta):
            self.theta = theta

        def __call__(self, x):
            mu = np.dot(x, self.theta)
            return np.random.normal(mu, 1)
        
    reward_fn = LinearNormalReward(REWARD_PARAM)
    num_actions_fn = lambda: args.num_actions
    # new
    optimal_reward_fn = functools.partial(
        optimal_reward, hidden_param=REWARD_PARAM
    )
    optimal_action_fn = functools.partial(
        optimal_action, hidden_param=REWARD_PARAM
    )

    suboptimal_arms_metric = tf_bandit_metrics.SuboptimalArmsMetric(
        optimal_action_fn
    )

    regret_metric = tf_bandit_metrics.RegretMetric(optimal_reward_fn)

    metrics = [regret_metric, suboptimal_arms_metric]
    
    # ====================================================
    # Define RL environment
    # ====================================================
    per_arm_py_env = p_a_env.StationaryStochasticPerArmPyEnvironment(
        global_context_sampling_fn = global_context_sampling_fn,
        arm_context_sampling_fn = per_arm_context_sampling_fn,
        max_num_actions = args.num_actions,
        reward_fn = reward_fn, # _get_rewards
        num_actions_fn = num_actions_fn,
        batch_size=args.batch_size
    )
    per_arm_tf_env = tf_py_environment.TFPyEnvironment(per_arm_py_env)
    
    logging.info('observation spec: ', per_arm_tf_env.observation_spec())
    logging.info('reward_spec: ', per_arm_tf_env.reward_spec())
    
    # ====================================================
    # cretae agent
    # ====================================================
    # with strategy.scope():
    train_step = tfa_train_utils.create_train_step()

    agent, network = _get_agent(
        agent_type=args.agent_type, 
        network_type=args.network_type, 
        environment=per_arm_tf_env
    )
        
    # ====================================================
    # train loop
    # ====================================================
    #start the timer and training
    start_time = time.time()

    metric_results = trainer_common.train_perarm(
        agent = agent,
        # replay_buffer = replay_buffer,
        # driver = driver,
        environment = per_arm_tf_env,
        # step_metric = step_metric,
        num_iterations = args.training_loops,
        steps_per_loop = args.steps_per_loop,
        log_interval = 1,
        # regret_metric = regret_metric,
        log_dir=log_dir,
        model_dir=args.artifacts_dir,
        root_dir=args.root_dir,
        async_steps_per_loop = args.async_steps_per_loop,
        resume_training_loops = False,
        get_replay_buffer_fn = None,
        get_training_loop_fn = None,
        training_data_spec_transformation_fn = None,
        additional_metrics = metrics
    )

    end_time = time.time()
    runtime_mins = int((end_time - start_time) / 60)
    print(f"complete train job in {runtime_mins} minutes")
    
    # ====================================================
    # log Vertex Experiments
    # ====================================================
    SESSION_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=3)) # handle restarts 
    
    if task_type == 'chief':
        logging.info(f" task_type logging experiments: {task_type}")
        logging.info(f" task_id logging experiments: {task_id}")
        logging.info(f" logging data to experiment run: {args.experiment_run}-{SESSION_id}")
        
        with vertex_ai.start_run(
            f'{args.experiment_run}-{SESSION_id}', 
            # tensorboard=args.tb_resource_name
        ) as my_run:
            
            logging.info(f"logging metrics...")
            # gather the metrics for the last epoch to be saved in metrics
            my_run.log_metrics(
                {
                    "AverageReturnMetric" : float(metric_results["AverageReturnMetric"][-1])
                    , "FinalRegretMetric" : float(metric_results["RegretMetric"][-1])
                }
            )

            logging.info(f"logging metaparams...")
            my_run.log_params(
                {
                    "agent_type": agent.name,
                    "network": network,
                    "runtime": runtime_mins,
                    "batch_size": args.batch_size, 
                    "training_loops": args.training_loops,
                    "steps_pre_loop": args.steps_per_loop,
                    # "rank_k": RANK_K,
                    "num_actions": args.num_actions,
                    "per_arm": str(PER_ARM),
                    "global_lyrs": str(data_config.GLOBAL_LAYERS),
                    "arm_lyrs": str(data_config.ARM_LAYERS),
                    "common_lyrs": str(data_config.COMMON_LAYERS),
                    "encoding_dim": data_config.ENCODING_DIM,
                    "eps_steps": data_config.EPS_PHASE_STEPS,
                }
            )

            vertex_ai.end_run()
            logging.info(f"EXPERIMENT RUN: {args.experiment_run}-{SESSION_id} has ended")
            
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
        project=project_number
        , location='us-central1'
        , experiment=args.experiment_name
    )
            
    execute_task(args = args)

if __name__ == "__main__":
    
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Python Version = %s", sys.version)
    logging.info("TensorFlow Version = %s", tf.__version__)
    # logging.info("TF_CONFIG = %s", os.environ.get("TF_CONFIG", "Not found"))
    # logging.info("DEVICES = %s", device_lib.list_local_devices())
    logging.info("Reinforcement learning task started...")
    
    main()
    
    logging.info("Reinforcement learning task completed.")