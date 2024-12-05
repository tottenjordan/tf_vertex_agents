
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.dsl import (
    component, 
    Metrics
)
from . import pipeline_config

@component(
    base_image=pipeline_config.DATA_PIPELINE_IMAGE,
    install_kfp_package=False
)
def write_trajectories_to_bq(
    project_id: str,
    location: str,
    pipeline_version: str,
    bq_dataset_name: str,
    bucket_name: str,
    example_gen_gcs_path: str,
    global_emb_size: int,
    mv_emb_size: int,
    num_oov_buckets: int,
    batch_size: int,
    dataset_size: int = 0,
    vocab_filename: str = "vocab_dict.pkl",
    is_testing: bool = False,
) -> NamedTuple('Outputs', [
    ('global_dim', int),
    ('per_arm_dim', int),
    ('tf_record_file', str),
    ('bq_table_ref', str),
    ('batch_size', int),
]):
    import os
    import json
    import logging
    import numpy as np
    import pickle as pkl
    from google.cloud import aiplatform, bigquery, storage
    from typing import Callable, Dict, List, Optional, TypeVar, Any
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # tf agents
    import tensorflow as tf
    from tf_agents import trajectories
    from tf_agents.trajectories import trajectory
    from tf_agents.bandits.policies import policy_utilities
    from tf_agents.bandits.specs import utils as bandit_spec_utils
    
    # this repo
    from src.networks import encoding_network as emb_features
    from src.data import data_utils as data_utils
    from src.data import data_config as data_config
    from src.utils import reward_factory as reward_factory
    from src.data_preprocessor import preprocess_utils

    # set client SDKs
    aiplatform.init(
        project=project_id,
        location=location,
        # experiment=experiment_name,
    )
    storage_client = storage.Client(project=project_id)
    bqclient = bigquery.Client(project=project_id)
    
    # set variables
    GCS_DATA_PATH        = f"gs://{bucket_name}/{example_gen_gcs_path}"
    NUM_GLOBAL_FEATURES  = len(data_utils.USER_FEATURE_NAMES)     # 6
    NUM_ARM_FEATURES     = len(data_utils.MOVIE_FEATURE_NAMES)    # 5
    EXPECTED_GLOBAL_DIM  = global_emb_size * NUM_GLOBAL_FEATURES
    EXPECTED_PER_ARM_DIM = mv_emb_size * NUM_ARM_FEATURES
    
    logging.info(f'GCS_DATA_PATH       : {GCS_DATA_PATH}')
    logging.info(f'NUM_GLOBAL_FEATURES : {NUM_GLOBAL_FEATURES}')
    logging.info(f'NUM_ARM_FEATURES    : {NUM_ARM_FEATURES}')
    logging.info(f'EXPECTED_GLOBAL_DIM : {EXPECTED_GLOBAL_DIM}')
    logging.info(f'EXPECTED_PER_ARM_DIM: {EXPECTED_PER_ARM_DIM}')

    # =========================================================
    # get data
    # =========================================================
    # download vocabs
    LOCAL_VOCAB_FILENAME = 'vocab_dict.pkl'
    print(f"Downloading vocab...")
    data_utils.download_blob(
        project_id = project_id,
        bucket_name = bucket_name, 
        source_blob_name = f'{example_gen_gcs_path}/vocabs/{vocab_filename}', 
        destination_file_name= LOCAL_VOCAB_FILENAME
    )
    filehandler = open(f"{LOCAL_VOCAB_FILENAME}", 'rb')
    vocab_dict = pkl.load(filehandler)
    filehandler.close()

    # get train and val examples
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO
    data_splits = ["train","val"]
    all_files = []
    
    for _split in data_splits:

        for blob in storage_client.list_blobs(
            f"{bucket_name}", 
            prefix=f'{example_gen_gcs_path}/{_split}'
        ):
            if '.tfrecord' in blob.name:
                all_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))
    print("Found these tfrecords:")
    print(all_files)

    if is_testing:
        all_files = all_files[:2]
        print(f"in testing mode; only using: {all_files}")
    dataset = tf.data.TFRecordDataset(all_files)
    dataset = dataset.map(data_utils._parse_function)
    
    # =========================================================
    # get emb dims
    # =========================================================
    print(f"getting embedding dimensions...")
    for i in range(1):
        iterator = iter(dataset.batch(1))
        data = next(iterator)
        
    embs = emb_features.EmbeddingModel(
        vocab_dict = vocab_dict,
        num_oov_buckets = num_oov_buckets,
        global_emb_size = global_emb_size,
        mv_emb_size = mv_emb_size,
        max_genre_length = data_config.MAX_GENRE_LENGTH,
    )
    test_globals = embs._get_global_context_features(data)
    test_arms = embs._get_per_arm_features(data)
    GLOBAL_DIM = test_globals.shape[1]            
    PER_ARM_DIM = test_arms.shape[1]
    print(f"GLOBAL_DIM  : {GLOBAL_DIM}")
    print(f"PER_ARM_DIM : {PER_ARM_DIM}")
    
    # =========================================================
    # trajectory function
    # =========================================================
    BQ_TMP_FILE   = "tmp_bq.json"
    BQ_TABLE_NAME = f"mv_b{batch_size}_g{global_emb_size}_a{mv_emb_size}_{pipeline_version}"
    BQ_TABLE_REF  = f"{project_id}.{bq_dataset_name}.{BQ_TABLE_NAME}"
    DS_GCS_DIR_PATH = f"gs://{bucket_name}/{example_gen_gcs_path}/{BQ_TABLE_NAME}"
    TFRECORD_FILE = f"{DS_GCS_DIR_PATH}/{BQ_TABLE_NAME}.tfrecord"
    
    print(f"BQ_TMP_FILE   : {BQ_TMP_FILE}")
    print(f"BQ_TABLE_NAME : {BQ_TABLE_NAME}")
    print(f"BQ_TABLE_REF  : {BQ_TABLE_REF}")
    print(f"DS_GCS_DIR_PATH : {DS_GCS_DIR_PATH}")
    print(f"TFRECORD_FILE : {TFRECORD_FILE}")
    
    # my trajectory functions
    def my_trajectory_fn(element):
        """Converts a dataset element into a trajectory."""
        global_features = embs._get_global_context_features(element)
        arm_features = embs._get_per_arm_features(element)

        observation = {
            bandit_spec_utils.GLOBAL_FEATURE_KEY: global_features
        }
        reward = reward_factory._get_rewards(element)

        policy_info = policy_utilities.PerArmPolicyInfo(
            chosen_arm_features=arm_features,
        )
        return trajectory.single_step(
            observation=observation,
            action=tf.zeros_like(
                reward, dtype=tf.int32
            ),
            policy_info=policy_info,
            reward=reward,
            discount=tf.zeros_like(reward)
        )

    # # calculate dataset_size
    # if not dataset_size:
    #     print(f"getting size of dataset...")
    #     dataset_size = dataset.reduce(0, lambda x,_: x+1).numpy()
    
    # write to local file
    print(f"writting trajectories to tmp file...")
    with open(BQ_TMP_FILE, "w") as f:
        for example in dataset.batch(batch_size, drop_remainder=True): #.take(count=dataset_size):
            _trajectories = my_trajectory_fn(example)
            _traj_dict = preprocess_utils.build_dict_from_trajectory(_trajectories)
            f.write(json.dumps(_traj_dict) + "\n")

    print(f"saving tmp file to: {example_gen_gcs_path}/{BQ_TABLE_NAME}/{BQ_TMP_FILE}")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{example_gen_gcs_path}/{BQ_TABLE_NAME}/{BQ_TMP_FILE}")
    blob.upload_from_filename(BQ_TMP_FILE)
    
    print(f"loading tmp file to bigquery...")
    with open(BQ_TMP_FILE, "rb") as source_file:
        load_job = bqclient.load_table_from_file(
            source_file, 
            BQ_TABLE_REF, 
            job_config=preprocess_utils.job_config
        )
    load_job.result() 
    
    # check table
    bq_table = bqclient.get_table(BQ_TABLE_REF)
    print(f"Got table: `{bq_table.project}.{bq_table.dataset_id}.{bq_table.table_id}`")
    print("Table has {} rows".format(bq_table.num_rows))

    return (
        GLOBAL_DIM,
        PER_ARM_DIM,
        TFRECORD_FILE,
        BQ_TABLE_REF,
        batch_size
    )
