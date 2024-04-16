import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp import dsl
from . import pipeline_config

@dsl.component(
    base_image=pipeline_config.POLICY_PIPE_IMAGE,
    install_kfp_package=False
)
def eval_agent_policy(
    project_id: str,
    location: str,
    pipeline_version: str,
    bucket_name: str,
    example_gen_gcs_path: str,
    # agent
    hparams: str,
    arftifacts_dir: str,
    # data
    ds_skip: int,
    ds_take: int,
    num_eval_samples: int, 
    total_eval_rewards: float,
    eval_ds: dsl.Input[dsl.Dataset],
    metrics: dsl.Output[dsl.Metrics]
):
    # imports
    import os
    import json
    import time
    import logging
    import numpy as np
    import pickle as pkl
    from pprint import pprint
    from google.cloud import aiplatform, storage
    from typing import Callable, Dict, List, Optional, TypeVar, Any
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # tf
    import tensorflow as tf
    from tf_agents.policies import py_tf_eager_policy
    
    # this repo
    from src.trainer import eval_perarm
    from src.data import data_utils
    
    # convert hparam dict
    HPARAMS = json.loads(hparams)
    pprint(HPARAMS)
    
    # set client SDKs
    aiplatform.init(
        project=project_id,
        location=location,
    )
    storage_client = storage.Client(project=project_id)
    # =========================================================
    # download vocabs
    # =========================================================
    LOCAL_VOCAB_FILENAME = 'vocab_dict.pkl'
    print(f"Downloading vocab...")
    data_utils.download_blob(
        project_id = project_id,
        bucket_name = bucket_name, 
        source_blob_name = f'{example_gen_gcs_path}/vocabs/{LOCAL_VOCAB_FILENAME}', 
        destination_file_name= LOCAL_VOCAB_FILENAME
    )
    filehandler = open(f"{LOCAL_VOCAB_FILENAME}", 'rb')
    vocab_dict = pkl.load(filehandler)
    filehandler.close()
    
    # =========================================================
    # get eval tf-records
    # =========================================================
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
        
    # =========================================================
    # load policy
    # =========================================================
    my_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        arftifacts_dir, load_specs_from_pbtxt=True
    )
    
    # =========================================================
    # run policy eval on val dataset
    # =========================================================
    print(f"evaluating loaded policy...")
    start_time = time.time()
    
    val_loss, preds, tr_rewards = eval_perarm._run_bandit_eval(
        policy = my_policy,
        data = eval_ds,
        eval_batch_size = HPARAMS['eval_batch_size'],
        per_arm_dim = HPARAMS['per_arm_dim'],
        global_dim = HPARAMS['global_dim'],
        vocab_dict = vocab_dict,
        num_oov_buckets = 1,
        global_emb_size = HPARAMS['global_emb_size'],
        mv_emb_size = HPARAMS['arm_emb_size'],
    )
    runtime_mins = int((time.time() - start_time) / 60)
    print(f"post-train val_loss     : {val_loss}")
    print(f"post-train eval runtime : {runtime_mins}")
    
    # =========================================================
    # log metrics
    # =========================================================
    total_pred_rewards = round(tf.reduce_sum(preds).numpy().tolist(), 2)
    reward_diff = round(abs(total_eval_rewards - total_pred_rewards), 2)
    avg_reward_vals = np.average([total_pred_rewards, total_eval_rewards])
    reward_percentage_diff = round((reward_diff / avg_reward_vals) * 100.0, 2)
    
    print(f"total_eval_rewards : {total_eval_rewards}")
    print(f"total_pred_rewards : {total_pred_rewards}")
    print(f"reward_diff        : {reward_diff}")
    print(f"avg_reward_vals    : {avg_reward_vals}")
    print(f"reward % diff      : {reward_percentage_diff}%")
    
    metrics.log_metric("total_eval_ds_rewards", total_eval_rewards)
    metrics.log_metric("total_predicted_rewards", total_pred_rewards)
    metrics.log_metric("reward_%_diff", reward_percentage_diff)
    metrics.log_metric("val_loss", round(val_loss.numpy().tolist(), 2))
