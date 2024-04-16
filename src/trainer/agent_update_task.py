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
from tf_agents.train.utils import strategy_utils

# logging
import logging
logging.disable(logging.WARNING)

from google.cloud import aiplatform
from google.cloud import storage

# tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# this repo
from . import train_batched_ds
# from src.data import data_utils, data_config
# from src.utils import train_utils, reward_factory
# from src.agents import agent_factory as agent_factory
# from src.networks import encoding_network as emb_features

# ====================================================
# Args
# ====================================================
def get_args(raw_args) -> argparse.Namespace: # : List[str]
    """
    TODO
    """
    parser = argparse.ArgumentParser()
    # Path parameters
    parser.add_argument("--project", default="hybrid-vertex", type=str)
    parser.add_argument("--location", default="us-central1", type=str)
    parser.add_argument("--bucket_name", default="tmp", type=str)
    parser.add_argument("--experiment_name", default="tmp-experiment", type=str)
    parser.add_argument("--experiment_run", default="tmp-experiment-run", type=str)
    parser.add_argument("--log_dir", default=None, type=str, help="Dir for TB logs")
    parser.add_argument("--artifacts_dir", type=str)
    parser.add_argument("--chkpoint_dir", default=None, type=str)
    # parser.add_argument("--hparams", required=True, type=json.loads)
    parser.add_argument("--hparams", required=True, type=str)
    # train job config
    parser.add_argument("--num_epochs", default=4, type=int, help="Number of epochs")
    parser.add_argument("--tf_record_file", default=None, type=str, help="gcs uri to batched tf-record")
    parser.add_argument("--log_interval", type=int, default=100, help="")
    parser.add_argument("--total_train_take", type=int, default=1000, help="")
    parser.add_argument("--total_train_skip", type=int, default=0, help="")
    # performance
    parser.add_argument('--tf_gpu_thread_count', type=str, required=False)
    parser.add_argument("--use_gpu", action='store_true', help="include for True; ommit for False")
    parser.add_argument("--use_tpu", action='store_true', help="include for True; ommit for False")
    parser.add_argument("--cache_train_data", action='store_true', help="include for True; ommit for False")

    return parser.parse_args(raw_args)

def main(args: argparse.Namespace):
    print("logging args....")
    print(args)
    print(f"hparams dict:")
    HPARAMS = json.loads(args.hparams)
    print(f"HPARAMS type: {type(HPARAMS)}")
    pprint(HPARAMS)
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
    from google.cloud import aiplatform
    from google.cloud import storage

    storage_client = storage.Client(project=args.project)
    aiplatform.init(
        project=args.project
        , location=args.location
        , experiment=args.experiment_name
    )
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
        print("All devices: ", tf.config.list_logical_devices('TPU'))

    print(f"distribution_strategy: {distribution_strategy}")
    if distribution_strategy == 'multiworker':
        task_type, task_id = (
            strategy.cluster_resolver.task_type,
            strategy.cluster_resolver.task_id
        )
    else:
        task_type, task_id = 'chief', None
    
    NUM_REPLICAS = distribution_strategy.num_replicas_in_sync
    print(f'NUM_REPLICAS = {NUM_REPLICAS}')
    print(f'task_type = {task_type}')
    print(f'task_id = {task_id}')
    # ====================================================
    # set Vertex AI env vars
    # ====================================================
    if 'AIP_TENSORBOARD_LOG_DIR' in os.environ:
        LOG_DIR=os.environ['AIP_TENSORBOARD_LOG_DIR']
        print(f'AIP_TENSORBOARD_LOG_DIR: {LOG_DIR}')
    else:
        LOG_DIR = args.log_dir
        print(f'LOG_DIR: {LOG_DIR}')
    print(f'TensorBoard log_dir: {LOG_DIR}')

    # [Do Not Change] Set the root directory for training artifacts.
    MODEL_DIR = os.environ["AIP_MODEL_DIR"]
    print(f'MODEL_DIR: {MODEL_DIR}')
    
    # ====================================================
    #start the timer and training
    # ====================================================
    start_time = time.time()

    train_loss, agent = train_batched_ds.train(
        hparams=HPARAMS,
        experiment_name=args.experiment_name,
        experiment_run=args.experiment_run,
        num_epochs = args.num_epochs,
        log_dir=LOG_DIR,
        artifacts_dir=args.artifacts_dir,
        chkpoint_dir=args.chkpoint_dir,
        tfrecord_file=args.tf_record_file, #TFRECORD_FILE,
        log_interval=args.log_interval,
        # chkpt_interval=10_000,
        use_gpu = args.use_gpu,
        use_tpu = args.use_tpu,
        # profiler = False,
        total_take = args.total_train_take,
        total_skip = args.total_train_skip,
        cache_train_data = args.cache_train_data,
    )
    end_time = time.time()
    runtime_mins = int((end_time - start_time) / 60)
    print(f"complete train job in {runtime_mins} minutes")
    
    # ====================================================
    # log Vertex Experiments
    # ====================================================
    if task_type == 'chief':
        print(f"task_type logging experiments: {task_type}")
        print(f"task_id logging experiments: {task_id}")
        print(f"logging data to experiment run: {args.experiment_run}")
        
        with aiplatform.start_run(
            f'{args.experiment_run}',
            # tensorboard=args.tb_resource_name
        ) as my_run:
            my_run.log_metrics(
                {
                    "train_loss" : round(float(train_loss[-1].tolist()),2)
                }
            )
            my_run.log_params(
                {
                    "agent_type": HPARAMS['agent_type'],
                    "network": HPARAMS['network_type'],
                    "global_dim": HPARAMS['global_dim'],
                    "per_arm_dim": HPARAMS['per_arm_dim'],
                    "runtime": runtime_mins,
                    "batch_size": HPARAMS['batch_size'], 
                    "global_lyrs": str(HPARAMS['global_layers']),
                    "arm_lyrs": str(HPARAMS['per_arm_layers']),
                    "common_lyrs": str(HPARAMS['common_layers']),
                    "arm_emb_size":HPARAMS['arm_emb_size'],
                    "global_emb_size": HPARAMS['global_emb_size'],
                }
            )
            aiplatform.end_run()
            tf.print(f"EXPERIMENT RUN: {args.experiment_run} has ended")

if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        level=logging.INFO, 
        datefmt='%d-%m-%y %H:%M:%S',
        stream=sys.stdout
    )
    print("Python Version = %s", sys.version)
    print("TensorFlow Version = %s", tf.__version__)
    print("Agent update starting...")

    args = get_args(sys.argv[1:])
    logging.info('Args: %s', args)

    main(args = args)

    print("Agent update complete")