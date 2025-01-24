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
import numpy as np
import pickle as pkl
from typing import List, Union

# logging
import logging
# logging.disable(logging.WARNING)

# tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# tensorflow
import tensorflow as tf
from tensorflow.python.client import device_lib

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

# google cloud
from google.cloud import aiplatform
from google.cloud import storage

# this repo
from . import offline_evaluation as offline_evaluation
from . import offline_metrics as offline_metrics

from src.data import data_utils as data_utils
from src.data import data_config as data_config
from src.networks import encoding_network as emb_features
from src.agents import topk_reinforce_agent as topk_reinforce_agent

from src.utils import rfa_utils as rfa_utils
from src.utils import train_utils as train_utils

# gcp clients
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
    
    # Hyperparameters
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--eval_batch_size", default=128, type=int)
    parser.add_argument("--num_eval_batches", default=0, type=int)
    
    # parser.add_argument("--num_epochs", default=4, type=int, help="Number of cycle through train data.")
    # parser.add_argument("--steps_per_loop", default=2, type=int)
    # parser.add_argument("--num_eval_steps", default=1000, type=int)
    # parser.add_argument("--num_actions", default=20, type=int, help="Number of actions (movie items) to choose from.")
    
    # agent and networks
    parser.add_argument("--input_embedding_size", type=int, default=100, help="")
    parser.add_argument('--input_fc_layer_params', type=str, required=False)
    parser.add_argument('--lstm_size', type=str, required=False)
    parser.add_argument('--output_fc_layer_params', type=str, required=False)
    parser.add_argument("--policy_num_actions", type=int, default=5, help="")
    parser.add_argument("--num_greedy_actions", type=int, default=4, help="")
    parser.add_argument("--sampled_softmax_num_negatives", type=int, default=20, help="")
    parser.add_argument("--off_policy_correction_exponent", type=int, default=16, help="")
    parser.add_argument("--use_supervised_loss_for_main_policy", action='store_true',help="ommit for False")
    parser.add_argument("--scann_num_candidate_actions", type=int, default=0, help="")
    parser.add_argument("--gamma", type=float, default=0.9, help="")
    parser.add_argument("--sum_grads_vars", action='store_true', help="ommit for False")
    parser.add_argument("--debug_summaries", action='store_true', help="ommit for False")
    
    # experiment & runs
    parser.add_argument("--experiment_name", default="tmp-experiment", type=str)
    parser.add_argument("--experiment_run", default="tmp-experiment-run", type=str)
    parser.add_argument("--tb_resource_name", default=None, type=str, help="")
    parser.add_argument("--log_vertex_experiment", action='store_true', help="ommit for False")

    # performance
    parser.add_argument("--use_gpu", action='store_true', help="include for True; ommit for False")
    parser.add_argument("--use_tpu", action='store_true', help="include for True; ommit for False")
    parser.add_argument('--tf_gpu_thread_count', default="1", type=str)
    parser.add_argument("--use_tf_functions", action='store_true', help="ommit for False")
    
    # train job
    parser.add_argument("--num_iterations", default=1000, type=int, help="Number of training steps.")
    parser.add_argument("--log_interval", type=int, default=1000, help="")
    parser.add_argument("--eval_interval", type=int, default=500, help="")
    parser.add_argument("--summary_interval", type=int, default=100, help="")
    parser.add_argument("--chkpt_interval", type=int, default=1000, help="")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="")
    
    # evaluation
    parser.add_argument('--eval_ks', type=str, required=False)

    return parser.parse_args(raw_args)

def main(args: argparse.Namespace):
    
    tf.print("logging args....")
    tf.print(args)

    # ====================================================
    # format args
    # ====================================================
    INPUT_FC_LAYER_PARAMS = train_utils.get_arch_from_string(args.input_fc_layer_params)
    LSTM_SIZE = train_utils.get_arch_from_string(args.lstm_size)
    OUTPUT_FC_LAYER_PARAMS = train_utils.get_arch_from_string(args.output_fc_layer_params)
    EVAL_Ks = train_utils.get_arch_from_string(args.eval_ks)
    
    tf.print(f'INPUT_FC_LAYER_PARAMS  : {INPUT_FC_LAYER_PARAMS}')
    tf.print(f'LSTM_SIZE              : {LSTM_SIZE}')
    tf.print(f'OUTPUT_FC_LAYER_PARAMS : {OUTPUT_FC_LAYER_PARAMS}')
    tf.print(f'EVAL_Ks                : {EVAL_Ks}')
    
    # =============================================
    # limiting GPU growth
    # =============================================
    if args.use_gpu:
        tf.print("limiting GPU growth....")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.print(f'detected: {len(gpus)} GPUs')
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                tf.print(e)

        # tf.debugging.set_log_device_placement(True) # logs all tf ops and their device placement;
        os.environ['TF_GPU_THREAD_MODE']='gpu_private'
        os.environ['TF_GPU_THREAD_COUNT']= f'{args.tf_gpu_thread_count}'
        os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
    # ====================================================
    # Set Device Strategy
    # ====================================================
    tf.print("Detecting devices....")
    tf.print("Setting device strategy...")
    
    # GPU - All variables and Agents need to be created under strategy.scope()
    if args.use_gpu:
        distribution_strategy = strategy_utils.get_strategy(tpu=args.use_tpu, use_gpu=args.use_gpu)
    
    if args.use_tpu:
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        distribution_strategy = tf.distribute.TPUStrategy(cluster_resolver)
        logging.info("All devices: ", tf.config.list_logical_devices('TPU'))

    if distribution_strategy == 'multiworker':
        task_type, task_id = (strategy.cluster_resolver.task_type, strategy.cluster_resolver.task_id)
    else:
        task_type, task_id = 'chief', None

    NUM_REPLICAS = distribution_strategy.num_replicas_in_sync
    tf.print(f"distribution_strategy: {distribution_strategy}")
    tf.print(f'NUM_REPLICAS: {NUM_REPLICAS}')
    tf.print(f'task_type: {task_type}')
    tf.print(f'task_id: {task_id}')
    
    # ====================================================
    # set Vertex AI env vars
    # ====================================================
    
    # Vertex tensorboard logging
    if 'AIP_TENSORBOARD_LOG_DIR' in os.environ:
        LOG_DIR=os.environ['AIP_TENSORBOARD_LOG_DIR']
        tf.print(f'LOG_DIR: {LOG_DIR}')
    else:
        LOG_DIR = args.log_dir
        tf.print(f'LOG_DIR: {LOG_DIR}')

    # Vertex Model registry
    if 'AIP_MODEL_DIR' in os.environ:
        MODEL_DIR=os.environ['AIP_MODEL_DIR']
        tf.print(f'AIP_MODEL_DIR: {MODEL_DIR}')
    else:
        MODEL_DIR = args.artifacts_dir
        tf.print(f'MODEL_DIR: {MODEL_DIR}')
    
    # =============================================
    # set GCP clients
    # =============================================
    
    aiplatform.init(
        project=data_config.PROJECT_ID, 
        location='us-central1',
        experiment=args.experiment_name,
        experiment_tensorboard=args.tb_resource_name
    )

    # ====================================================
    # train & val files
    # ====================================================
    EXAMPLE_GEN_GCS_PATH = data_config.EXAMPLE_GEN_GCS_PATH
    GCS_DATA_PATH = f"gs://{args.bucket_name}/{EXAMPLE_GEN_GCS_PATH}"
    tf.print(f"GCS_DATA_PATH: {GCS_DATA_PATH}")
    
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

    # TODO: jt optimize and clean
    tf.print(f'decoding vocab dict...')
    vocab_dict_decoded = [z.decode("utf-8") for z in VOCAB_DICT['movie_id']]
    vocab_dict_decoded.remove("UNK")
    vocab_dict_decoded = tf.strings.to_number(
        vocab_dict_decoded,
        out_type=tf.dtypes.int64,
        name=None
    )
    vocab_dict_decoded = vocab_dict_decoded.numpy()
    VOCAB_DICT['movie_id_int'] = vocab_dict_decoded
    
    # ====================================================
    # summary writers
    # ====================================================
    # _train_log_dir = os.path.join(LOG_DIR, 'train')
    # _eval_log_dir = os.path.join(LOG_DIR, 'eval')
    # tf.print(f'train_log_dir : {_train_log_dir}')
    # tf.print(f'eval_log_dir  : {_eval_log_dir}')
    
    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        LOG_DIR, flush_millis=10 * 1000)
    train_summary_writer.set_as_default()
    
    # eval_summary_writer = tf.compat.v2.summary.create_file_writer(
    #     _eval_log_dir, flush_millis=10 * 1000)
    
    global_step = tf.compat.v1.train.get_or_create_global_step()
    tf.print(f'global_step  : {global_step.numpy()}')
    # ====================================================
    # create agent, networks, layers, and variables
    # ====================================================
    with tf.compat.v2.summary.record_if(
        lambda: tf.math.equal(global_step % args.summary_interval, 0)):

        action_lookup_layer = tf.keras.layers.IntegerLookup(
            vocabulary=VOCAB_DICT['movie_id_int'], mask_value=None
        )
        inverse_action_lookup_layer = tf.keras.layers.IntegerLookup(
            vocabulary=action_lookup_layer.get_vocabulary(), 
            mask_value=None,
            invert=True
        )
        observation_lookup_layer = tf.keras.layers.IntegerLookup(
            vocabulary=VOCAB_DICT['movie_id_int'], 
            mask_value=None
        )
        
        action_vocab_size = action_lookup_layer.vocab_size()
        obs_vocab_size = observation_lookup_layer.vocab_size()
        tf.print(f"action_vocab_size : {action_vocab_size}")
        tf.print(f"obs_vocab_size    : {obs_vocab_size}")

        # Note that the minimum in the spec is 0 and the maximum is bound inclusive.
        # So setting maximum = vocab_size - 1 accounts for all actions in the
        # vocabulary, including OOV items.
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
        # Agent
        # ====================================================
        if args.off_policy_correction_exponent < 1:
            OFF_POLICY_CORRECTION_EXPONENT=None
        else:
            OFF_POLICY_CORRECTION_EXPONENT = args.off_policy_correction_exponent

        if args.scann_num_candidate_actions < 1:
            SCANN_NUM_CANDIDATE_ACTIONS=None
        else:
            SCANN_NUM_CANDIDATE_ACTIONS = args.scann_num_candidate_actions

        if args.sampled_softmax_num_negatives < 1:
            SAMPLED_SOFTMAX_NUM_NEGATIVES=None
        else:
            SAMPLED_SOFTMAX_NUM_NEGATIVES = args.sampled_softmax_num_negatives
            
        # with distribution_strategy.scope():
        state_embedding_network = rfa_utils.create_state_embedding_network(
            observation_lookup_layer=observation_lookup_layer,
            input_embedding_size=args.input_embedding_size,
            input_fc_layer_units=INPUT_FC_LAYER_PARAMS,
            lstm_size=LSTM_SIZE,
            output_fc_layer_units=OUTPUT_FC_LAYER_PARAMS
        )
        tf_agent = topk_reinforce_agent.TopKOffPolicyReinforceAgent(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            state_embedding_network=state_embedding_network,
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            off_policy_correction_exponent=OFF_POLICY_CORRECTION_EXPONENT,
            action_lookup_layer=action_lookup_layer,
            inverse_action_lookup_layer=inverse_action_lookup_layer,
            policy_num_actions=args.policy_num_actions,
            use_supervised_loss_for_main_policy=args.use_supervised_loss_for_main_policy,
            num_candidate_actions=SCANN_NUM_CANDIDATE_ACTIONS,
            num_greedy_actions=args.policy_num_actions,
            sampled_softmax_num_negatives=SAMPLED_SOFTMAX_NUM_NEGATIVES,
            train_step_counter=global_step,
            gamma=args.gamma,
            summarize_grads_and_vars=args.sum_grads_vars,
            debug_summaries=False, # TODO: error with summary stats
            name='TopKOffPolicyReinforceAgent'
        )
        tf_agent.initialize()
        # ====================================================
        # metrics and checkpoints
        # ====================================================
        def non_oov_filter(trajectory):
            """Returns True for non OOV actions."""
            return action_lookup_layer(trajectory.action) != 0

        offline_eval_metrics = [
            # offline_metrics.AccuracyAtK(),
            # offline_metrics.AveragePerClassAccuracyAtK(
            #     vocabulary_size=action_vocab_size,
            #     action_lookup=action_lookup_layer
            # ),
            # WeightedReturns requires action logits to work, make sure
            # emit_logits_as_info = True in the TopKOffPolicyReinforcePolicy
            offline_metrics.WeightedReturns(
                gamma=1.,
                action_lookup=action_lookup_layer,
                weight_by_probabilities=True,
                name='WeightedReturnsProb_gamma_1'
            ),
            offline_metrics.WeightedReturns(
                gamma=0.6,
                action_lookup=action_lookup_layer,
                weight_by_probabilities=True,
                name='WeightedReturnsProb_gamma_p6'
            ),
            offline_metrics.WeightedReturns(
                gamma=0.4,
                action_lookup=action_lookup_layer,
                weight_by_probabilities=True,
                name='WeightedReturnsProb_gamma_p4'
            ),
            offline_metrics.WeightedReturns(
                gamma=0,
                action_lookup=action_lookup_layer,
                weight_by_probabilities=True,
                name='WeightedReturnsProb_gamma_0'
            ),
            # LogProb
            offline_metrics.WeightedReturns(
                gamma=1.,
                action_lookup=action_lookup_layer,
                weight_by_probabilities=False,
                name='WeightedReturnsLogProb_gamma_1'
            ),
            offline_metrics.WeightedReturns(
                gamma=0.6,
                action_lookup=action_lookup_layer,
                weight_by_probabilities=False,
                name='WeightedReturnsLogProb_gamma_p6'
            ),
            offline_metrics.WeightedReturns(
                gamma=0.4,
                action_lookup=action_lookup_layer,
                weight_by_probabilities=False,
                name='WeightedReturnsLogProb_gamma_p4'
            ),
            offline_metrics.WeightedReturns(
                gamma=0,
                action_lookup=action_lookup_layer,
                weight_by_probabilities=False,
                name='WeightedReturnsLogProb_gamma_0'
            ),
        ]
        # TODO: paramterize w/ args.policy_num_actions
        for k in EVAL_Ks: #[1, 5, 10]:
            offline_eval_metrics.append(
                offline_metrics.AccuracyAtK(
                    trajectory_filter=non_oov_filter,
                    name='AccuracyAtK_' + str(k),
                    k=k
                )
            )
            offline_eval_metrics.append(
                offline_metrics.AveragePerClassAccuracyAtK(
                    vocabulary_size=action_vocab_size,
                    action_lookup=action_lookup_layer,
                    trajectory_filter=non_oov_filter,
                    name='AveragePerClassAccuracyAtK_' + str(k),
                    k=k
                )
            )
            offline_eval_metrics.append(
                offline_metrics.LastActionAccuracyAtK(
                    trajectory_filter=non_oov_filter,
                    name='LastActionAccuracyAtK_' + str(k),
                    k=k
                )
            )        
        
        tf.print(f"setting train_checkpointer: {args.chkpoint_dir}")
        train_checkpointer = train_utils.restore_and_get_checkpoint_manager(
            root_dir=args.chkpoint_dir, 
            agent=tf_agent, 
            metrics=offline_eval_metrics, 
            step_metric=global_step
        )
        POLICY_CHEKPT_DIR = os.path.join(args.chkpoint_dir, 'policy')
        tf.print(f"setting policy_checkpointer: {POLICY_CHEKPT_DIR}")
        policy_checkpointer = common.Checkpointer(
            ckpt_dir=POLICY_CHEKPT_DIR,
            policy=tf_agent.policy,
            global_step=global_step
        )
        # tf.print(f"agent.train_step_counter: {tf_agent.train_step_counter.value().numpy()}")
        # ====================================================
        # saver
        # ====================================================
        topk_policy = tf_agent.policy
        _time_step_spec_with_time_dim = tf.nest.map_structure(
            lambda spec: policy_saver.add_batch_dim(spec, [None]),
            topk_policy.time_step_spec,
        )
        _input_fn_and_spec = (
            lambda x: x,
            (_time_step_spec_with_time_dim, topk_policy.policy_state_spec),
        )
        saver = policy_saver.PolicySaver(
            policy = topk_policy, 
            train_step = global_step,
            input_fn_and_spec = _input_fn_and_spec,
        )
        tf.print(f"saver signatures:")
        for key in saver.signatures.keys():
            tf.print(f"{key} : {saver.signatures[key]}")
        
        # datasets
        _process_example_fn = functools.partial(
            rfa_utils.example_proto_to_trajectory,
            sequence_length=data_config.MAX_CONTEXT_LENGTH
        )
        train_dataset = rfa_utils.create_tfrecord_ds(
            tf.io.gfile.glob(train_files),
            num_shards=len(train_files),
            process_example_fn=_process_example_fn,
            batch_size=args.batch_size
        )
        train_dataset_iterator = iter(train_dataset)
        
        eval_dataset = rfa_utils.create_tfrecord_ds(
            tf.io.gfile.glob(val_files),
            process_example_fn=_process_example_fn,
            batch_size=args.eval_batch_size,
            num_shards=len(val_files),
            repeat=False,
            drop_remainder=True
        )
        if args.num_eval_batches > 0:
            eval_dataset = eval_dataset.take(args.num_eval_batches)
        tf.print(f"eval data size: {len(list(eval_dataset))}\n")
        
        if args.use_tf_functions:
            tf_agent.train = common.function(tf_agent.train)
            
        def _train_step():
            trajectory, weights = next(train_dataset_iterator)
            return tf_agent.train(trajectory, weights).loss

        if args.use_tf_functions:
            _train_step = common.function(_train_step)

        # ====================================================
        # Training loop
        # ====================================================
        tf.print(f"starting train loop...")
        
        TENSORBOARD_ID = args.tb_resource_name.split('/')[-1]
        tf.print(f'TENSORBOARD_ID: {TENSORBOARD_ID}')

        # Continuous monitoring w/ TensorBoard
        aiplatform.start_upload_tb_log(
            tensorboard_id=TENSORBOARD_ID,
            tensorboard_experiment_name=args.experiment_name,
            logdir=LOG_DIR,
            experiment_display_name=args.experiment_name,
            run_name_prefix=f"{args.experiment_run}-",
        )

        try:
        
            list_o_loss=[]
            time_acc = 0
            timed_at_step = global_step.numpy()
            start_train = time.time()

            for i in range(args.num_iterations):
                train_loss = _train_step()
                list_o_loss.append(train_loss.numpy())

                if global_step.numpy() % args.log_interval == 0:
                    tf.print(f'step = {global_step.numpy()}: loss = {train_loss.numpy()}')

                if global_step.numpy() % args.eval_interval == 0:
                    time_acc += time.time() - start_train
                    steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
                    tf.print(f"\n{round(steps_per_sec,3)} steps/sec")
                    timed_at_step = global_step.numpy()
                    time_acc = 0

                    tf.print(f"\nEval at step: {global_step.numpy()}...")
                    
                    # TODO: uncomment for ScaNN layer
                    # tf_agent.post_process_policy()

                    offline_evaluation.evaluate(
                        tf_agent.policy,
                        eval_dataset,
                        offline_eval_metrics=offline_eval_metrics,
                        train_step=global_step,
                        summary_writer=train_summary_writer, # eval_summary_writer,
                        summary_prefix='Metrics',
                    )
                    metric_utils.log_metrics(offline_eval_metrics)
                    for metric in offline_eval_metrics:
                        tf.print(metric.name, ' = ', metric.result().numpy())
                    tf.print("="*40 + "\n")

                if global_step.numpy() % args.chkpt_interval == 0:
                    tf.print(f"global_step for checkpoints: {global_step.numpy()}")
                    
                    train_checkpointer.save(global_step)
                    tf.print(f"saved train_checkpointer to {args.chkpoint_dir}")

                    # TODO: uncomment for ScaNN layer
                    # tf.print(f"post processing policy...")
                    # tf_agent.post_process_policy()

                    policy_checkpointer.save(global_step)
                    tf.print(f"saved policy_checkpointer to {POLICY_CHEKPT_DIR}")
        finally:
            aiplatform.end_upload_tb_log()
        
    runtime_mins = int((time.time() - start_train) / 60)
    tf.print(f"completed train job in {runtime_mins} minutes")
    
    train_checkpointer.save(global_step)
    tf.print(f"saved train_checkpointer to: {args.chkpoint_dir}")

    # TODO: uncomment for ScaNN layer
    # tf.print(f"post processing policy...")
    # tf_agent.post_process_policy()
    
    policy_checkpointer.save(global_step)
    tf.print(f"saved policy_checkpointer to: {POLICY_CHEKPT_DIR}")

    saver.save(MODEL_DIR)
    tf.print(f"saved trained policy to: {MODEL_DIR}")
    
    # ====================================================
    # log Vertex Experiments
    # ====================================================
    if args.log_vertex_experiment:
        
        # log experiment to Vertex
        metric_dict = {}
        for met in offline_eval_metrics:
            metric_dict[met.name] = round(float(met.result().numpy()), 5)

        exp_params = {
            "off_policy_exp" : int(args.off_policy_correction_exponent),
            "use_sl_loss" : str(args.use_supervised_loss_for_main_policy),
            "num_actions" : int(args.policy_num_actions),
            "num_greedy_actions" : int(args.num_greedy_actions),
            "softmax_num_negatives" : int(args.sampled_softmax_num_negatives),
            "gamma": float(args.gamma),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
        }

        if task_type == 'chief':
            tf.print(f" task_type logging experiments: {task_type}")
            tf.print(f" task_id logging experiments: {task_id}")
            tf.print(f" logging data to experiment run: {args.experiment_run}")
            
            # aiplatform.start_run(args.experiment_run)
            # aiplatform.log_params(exp_params)
            # aiplatform.log_metrics(metric_dict)
            # aiplatform.end_run()
            
            tf.print(f"EXPERIMENT RUN: {args.experiment_run} has ended")

            with aiplatform.start_run(args.experiment_run) as my_run:
                # tensorboard=args.tb_resource_name
                tf.print(f"logging params & metrics...")
                my_run.log_params(exp_params)
                my_run.log_metrics(metric_dict)

                aiplatform.end_run()
                tf.print(f"EXPERIMENT RUN: {args.experiment_run} has ended")
            
if __name__ == "__main__":
    # logging.set_verbosity(logging.INFO)
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