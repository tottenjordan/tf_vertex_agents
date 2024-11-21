"""
The entrypoint for training a REINFORCE Recommender Agent.
"""
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

# tensorflow
import tensorflow as tf

# tf-agents
from tf_agents.utils import common
from tf_agents.specs import tensor_spec
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.metrics import export_utils
from tf_agents.policies import policy_saver
from tf_agents.trajectories import trajectory
from tf_agents.train.utils import strategy_utils
from tf_agents.policies import py_tf_eager_policy
from tf_agents.trajectories import time_step as ts

from tf_agents.replay_buffers import tf_uniform_replay_buffer

# this repo
from src.data import data_utils as data_utils
from src.data import data_config as data_config
from src.agents import rfa_utils as rfa_utils
from src.networks import encoding_network as emb_features
from src.agents import topk_reinforce_agent as topk_reinforce_agent
from src.agents import offline_evaluation as offline_evaluation
from src.agents import offline_metrics as offline_metrics

# clients
storage_client = storage.Client(project=data_config.PROJECT_ID)

# ====================================================
# get train & val datasets
# ====================================================
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO

def get_filenames(
    bucket_name,
    prefix,
    split,
):
    file_list = []
    for blob in storage_client.list_blobs(
        f"{bucket_name}", 
        prefix=f'{prefix}/{split}'
    ):
        if '.tfrecord' in blob.name:
            file_list.append(
                blob.public_url.replace(
                    "https://storage.googleapis.com/", "gs://"
                )
            )
    
    return file_list

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
    
    # experiment & runs
    parser.add_argument("--experiment_name", default="tmp-experiment", type=str)
    parser.add_argument("--experiment_run", default="tmp-experiment-run", type=str)

    # performance
    parser.add_argument("--use_gpu", action='store_true', help="include for True; ommit for False")
    parser.add_argument("--use_tpu", action='store_true', help="include for True; ommit for False")
    
    return parser.parse_args(raw_args)

def main(args: argparse.Namespace):
    
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
    # ====================================================
    # Set Device Strategy
    # ====================================================
    print("Detecting devices....")
    print("Setting device strategy...")
    # GPU - All variables and Agents need to be created under strategy.scope()
    if args.use_gpu:
        distribution_strategy = strategy_utils.get_strategy(tpu=args.use_tpu, use_gpu=args.use_gpu)
        # distribution_strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

    # if args.use_tpu:
    #     cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    #     tf.config.experimental_connect_to_cluster(cluster_resolver)
    #     tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    #     distribution_strategy = tf.distribute.TPUStrategy(cluster_resolver)
    #     logging.info("All devices: ", tf.config.list_logical_devices('TPU'))

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
    
    # =============================================
    # set GCP clients
    # =============================================
    from google.cloud import aiplatform as vertex_ai
    from google.cloud import storage

    storage_client = storage.Client(project=data_config.PROJECT_ID)
    
    vertex_ai.init(
        project=data_config.PROJECT_ID
        , location='us-central1'
        , experiment=args.experiment_name
    )
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
    
    train_log_dir = os.path.join(log_dir, 'train')
    eval_log_dir = os.path.join(log_dir, 'eval')
    tf.print(f'train_log_dir : {train_log_dir}')
    tf.print(f'eval_log_dir  : {eval_log_dir}')
    # ====================================================
    # train & val files
    # ====================================================
    EXAMPLE_GEN_GCS_PATH = data_config.EXAMPLE_GEN_GCS_PATH
    GCS_DATA_PATH = f'gs://{args.bucket_name}/{EXAMPLE_GEN_GCS_PATH}"
    tf.print(f'GCS_DATA_PATH: {GCS_DATA_PATH}')
    
    train_files = get_filenames(
        bucket_name=args.bucket_name,
        prefix=EXAMPLE_GEN_GCS_PATH,
        split="train",
    )
    val_files = get_filenames(
        bucket_name=args.bucket_name,
        prefix=EXAMPLE_GEN_GCS_PATH,
        split="val",
    )
    # ====================================================
    # get action vocab
    # ====================================================
    # TODO: parameterize
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
    
    # update vocab dict
    # TODO: jt optimize and clean
    vocab_dict_decoded = [z.decode("utf-8") for z in vocab_dict['movie_id']]
    vocab_dict_decoded.remove("UNK")
    vocab_dict_decoded = tf.strings.to_number(
        vocab_dict_decoded,
        out_type=tf.dtypes.int64,
        name=None
    )
    vocab_dict_decoded = vocab_dict_decoded.numpy()
    vocab_dict['movie_id_int'] = vocab_dict_decoded
    # ====================================================
    # lookup layers
    # ====================================================
    # TODO: parameterize
    action_lookup_layer = tf.keras.layers.IntegerLookup(
        vocabulary=vocab_dict['movie_id_int'], 
        mask_value=None
    )
    action_vocab_size = action_lookup_layer.vocab_size() # 3885

    inverse_action_lookup_layer = tf.keras.layers.IntegerLookup(
        vocabulary=action_lookup_layer.get_vocabulary(), 
        mask_value=None,
        invert=True
    )
    # observations are just past actions:
    observation_lookup_layer = tf.keras.layers.IntegerLookup(
        vocabulary=vocab_dict['movie_id_int'], 
        mask_value=None
    )
    obs_vocab_size = observation_lookup_layer.vocab_size()
    # ====================================================
    # tensor specs
    # ====================================================
    # TODO: parameterize
    observation_spec = tensor_spec.BoundedTensorSpec(
        shape=[],
        dtype=tf.int64, # tf.string | tf.int64,
        minimum=0,
        maximum=action_vocab_size - 1,
        name='observation'
    )
    time_step_spec = ts.time_step_spec(observation_spec=observation_spec)

    action_spec = tensor_spec.BoundedTensorSpec(
        shape=[],
        dtype=tf.int64, # tf.string | tf.int64,
        minimum=0,
        maximum=action_vocab_size - 1,
        name='action'
    )
    # ====================================================
    # Network 
    # ====================================================
    # TODO: parameterize
    input_embedding_size=100
    input_fc_layer_params=(100, 100)
    lstm_size=(25,)
    output_fc_layer_params=(10,)

    state_embedding_network = rfa_utils.create_state_embedding_network(
        observation_lookup_layer=observation_lookup_layer,
        input_embedding_size=input_embedding_size,
        input_fc_layer_units=input_fc_layer_params,
        lstm_size=lstm_size,
        output_fc_layer_units=output_fc_layer_params
    )
    # ====================================================
    # Agent
    # ====================================================
    # TODO: parameterize
    learning_rate = 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    global_step = tf.compat.v1.train.get_or_create_global_step()
    policy_num_actions=5
    num_greedy_actions=4
    scann_num_candidate_actions=None
    sampled_softmax_num_negatives=None
    off_policy_correction_exponent=None
    use_supervised_loss_for_main_policy=False
    GAMMA=0.9
    SUMMARIZE_GRADS_AND_VARS=True
    DEBUG_SUMMARIES=False # TODO: error with summary stats
    
    with distribution_strategy.scope():
        
        global_step = tf.compat.v1.train.get_or_create_global_step()

        tf_agent = topk_reinforce_agent.TopKOffPolicyReinforceAgent(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            state_embedding_network=state_embedding_network,
            optimizer=optimizer,
            off_policy_correction_exponent=off_policy_correction_exponent,
            action_lookup_layer=action_lookup_layer,                  # action_lookup_layer | None
            inverse_action_lookup_layer=inverse_action_lookup_layer,  # inverse_action_lookup_layer | None
            policy_num_actions=policy_num_actions,
            use_supervised_loss_for_main_policy=use_supervised_loss_for_main_policy,
            num_candidate_actions=scann_num_candidate_actions,
            num_greedy_actions=policy_num_actions,
            sampled_softmax_num_negatives=sampled_softmax_num_negatives,
            train_step_counter=global_step,
            gamma=GAMMA,
            summarize_grads_and_vars=SUMMARIZE_GRADS_AND_VARS,
            debug_summaries=DEBUG_SUMMARIES,
            name='TopKOffPolicyReinforceAgent'
        )

        tf_agent.initialize()

    train_checkpointer = common.Checkpointer(
        ckpt_dir=args.chkpoint_dir,
        agent=tf_agent,
        global_step=global_step
    )

    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(args.chkpoint_dir, 'policy'),
        policy=tf_agent.policy,
        global_step=global_step
    )
    
    # policy checkpoints
    train_checkpointer.initialize_or_restore()
    evaluate = offline_evaluation.evaluate
    
    # summary writers
    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        train_log_dir, flush_millis=10 * 1000
    )
    train_summary_writer.set_as_default()
    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        eval_log_dir, flush_millis=10 * 1000
    )
    # ====================================================
    # create datasets
    # ====================================================
    # TODO: parameterize
    
    use_tf_functions= True
    sequence_length=10
    train_batch_size=64 # 64 | 5
    eval_batch_size=64 # 64 | 5
    num_eval_batches=10
    
    # TODO: common?
    tf_agent.train = common.function(tf_agent.train)
    
    process_example_fn = functools.partial(
        example_proto_to_trajectory,
        sequence_length=sequence_length
    )

    train_dataset = create_tfrecord_ds(
        tf.io.gfile.glob(train_files),
        num_shards=len(train_files),
        process_example_fn=process_example_fn,
        batch_size=train_batch_size
    )
    train_dataset_iterator = iter(train_dataset)

    eval_dataset = create_tfrecord_ds(
        tf.io.gfile.glob(val_files),
        process_example_fn=process_example_fn,
        batch_size=eval_batch_size,
        num_shards=len(val_files),
        repeat=False,
        drop_remainder=True
    )
    if num_eval_batches is not None:
        eval_dataset = eval_dataset.take(num_eval_batches)
    # ====================================================
    # metrics
    # ====================================================
    # TODO: parameterize
    from tf_agents.eval import metric_utils
    from tf_agents.metrics import export_utils

    def _export_metrics_and_summaries(step, metrics):
        """Exports metrics and tf summaries."""
        metric_utils.log_metrics(metrics)
        export_utils.export_metrics(step=step, metrics=metrics)
        for metric in metrics:
            metric.tf_summaries(train_step=step)

    offline_eval_metrics = [
        offline_metrics.AccuracyAtK(),
        offline_metrics.AveragePerClassAccuracyAtK(
            action_vocab_size, action_lookup=action_lookup_layer
        ),
        offline_metrics.WeightedReturns(
            gamma=1.,
            action_lookup=action_lookup_layer,
            name='WeightedReturns_gamma_1'
        ),
        # offline_metrics.LastActionAccuracyAtK(),
    ]
    # ====================================================
    # XXXX
    # ====================================================
    # TODO: parameterize
    num_iterations=10000

    eval_interval=num_iterations/4 # 2500
    log_interval=num_iterations/10 # 1000
    summary_interval=100

    train_checkpoint_interval=15000
    policy_checkpoint_interval=15000
    
    with distribution_strategy.scope():
        
        # here
        list_o_loss=[]
        start_time = time.time()

        with tf.compat.v2.summary.record_if(
            lambda: tf.math.equal(global_step % summary_interval, 0)
        ):
            timed_at_step = global_step.numpy()
            time_acc = 0

            for _ in range(num_iterations):
                start_time = time.time()

                traj_ex, weights_ex = next(train_dataset_iterator)
                train_loss = tf_agent.train(
                    experience=traj_ex, weights=weights_ex
                )
                list_o_loss.append(train_loss.loss.numpy())
                time_acc += time.time() - start_time

                global_step_val = global_step.numpy()
                if global_step_val % log_interval == 0:
                    print(f"step = {global_step_val}, loss = {train_loss.loss}")
                    steps_per_sec = (global_step_val - timed_at_step) / time_acc

                    print(f"{round(steps_per_sec,3)} steps/sec")
                    tf.summary.scalar(
                        name='global_steps_per_sec', 
                        data=steps_per_sec, 
                        step=global_step
                    )

                    timed_at_step = global_step_val
                    time_acc = 0

                if global_step_val % train_checkpoint_interval == 0:
                    train_checkpointer.save(global_step=global_step_val)

                if global_step_val % policy_checkpoint_interval == 0:
                    tf_agent.post_process_policy()
                    policy_checkpointer.save(global_step=global_step_val)

                if global_step_val % eval_interval == 0:
                    tf_agent.post_process_policy()
                    print(f"Eval at step: {global_step_val}")
                    evaluate(
                        tf_agent.policy,
                        eval_dataset,
                        offline_eval_metrics=offline_eval_metrics,
                        train_step=global_step,
                        summary_writer=eval_summary_writer,
                        summary_prefix='Metrics',
                    )
                    metric_utils.log_metrics(offline_eval_metrics)

        runtime_mins = int((time.time() - start_time) / 60)
        print(f"total runtime : {runtime_mins}")

        for metric in offline_eval_metrics:
            print(f"\nOffline eval metrics:")
            print(metric.name, ' = ', metric.result().numpy())