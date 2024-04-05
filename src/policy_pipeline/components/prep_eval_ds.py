import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp import dsl
from . import pipeline_config

@dsl.component(
    base_image=pipeline_config.POLICY_PIPE_IMAGE,
    install_kfp_package=False
)
def prep_eval_ds(
    project_id: str,
    location: str,
    pipeline_version: str,
    bucket_name: str,
    example_gen_gcs_path: str,
    eval_ds: dsl.Output[dsl.Dataset],
    ds_skip: int = 0,
    ds_take: int = 0,
) -> NamedTuple('Outputs', [
    ('num_eval_samples', int),
    ('total_eval_rewards', float),
    ('ds_skip', int),
    ('ds_take', int),
]):
    
    import os
    import json
    import logging
    import numpy as np
    import pickle as pkl
    from google.cloud import aiplatform, bigquery, storage
    from typing import Callable, Dict, List, Optional, TypeVar, Any
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    import tensorflow as tf
    
    # this repo
    from src.data import data_utils
    
    # set client SDKs
    aiplatform.init(
        project=project_id,
        location=location,
        # experiment=experiment_name,
    )
    storage_client = storage.Client(project=project_id)
    
    # get eval tf-records
    val_files = []
    for blob in storage_client.list_blobs(f"{bucket_name}", prefix=f'{example_gen_gcs_path}/val'):
        if '.tfrecord' in blob.name:
            val_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))
            
    val_dataset = tf.data.TFRecordDataset(val_files)
    val_dataset = val_dataset.map(data_utils._parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    
    eval_ds = val_dataset.batch(1)
    
    if ds_skip > 0:
        eval_ds = eval_ds.skip(ds_skip)
        logging.info(f"setting dataset skip: {ds_skip}")
    
    if ds_take > 0:
        eval_ds = eval_ds.take(ds_take)
        logging.info(f"setting dataset take: {ds_take}")
        
    # get length (size) of eval ds
    NUM_EVAL_SAMPLES = len(list(eval_ds))
    logging.info(f"NUM_EVAL_SAMPLES : {NUM_EVAL_SAMPLES}")
    
    # get total rewards from eval slice
    val_rewards = []
    for x in eval_ds:
        val_rewards.append(x[f"{data_utils.TARGET_FEATURE_NAME}"][0].numpy())
    
    TOTAL_EVAL_REWARD = tf.reduce_sum(val_rewards).numpy().tolist()
    logging.info(f"TOTAL_EVAL_REWARD : {TOTAL_EVAL_REWARD}")
    
    return (
        NUM_EVAL_SAMPLES,
        TOTAL_EVAL_REWARD,
        ds_skip,
        ds_take,
    )
