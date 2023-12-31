"""Prediction server that uses a trained policy to give predicted actions."""
import os
import pickle as pkl

# fastapi
from fastapi.logger import logger
from fastapi import FastAPI, Request

# tensorflow
import tensorflow as tf
import tf_agents
from tf_agents.policies import py_tf_eager_policy
from tf_agents.trajectories import time_step as ts

# GCP
from google.cloud import storage

# this repo
from . import data_config
from . import emb_features_pred as emb_features

# import sys
# sys.path.append("..")
# from src.perarm_features import emb_features as emb_features
# from src.perarm_features import reward_factory as reward_factory

import logging

# ====================================================
# logging
# ====================================================
gunicorn_logger = logging.getLogger('gunicorn.error')
logger.handlers = gunicorn_logger.handlers

if __name__ != "main":
    logger.setLevel(gunicorn_logger.level)
else:
    logger.setLevel(logging.DEBUG)

# ====================================================
# helper functions
# ====================================================
def download_blob(project_id, bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client(project=data_config.PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )
    
# get trajectory step for prediction
def _get_pred_step(
    feature, 
    reward_np
):
    
    infer_step = ts.TimeStep(
        tf.constant(
            ts.StepType.FIRST, 
            dtype=tf.int32, 
            shape=[],
            name='step_type'
        ),
        tf.constant(
            reward_np, dtype=tf.float32, shape=[], name='reward'
        ),
        tf.constant(
            1.0, dtype=tf.float32, shape=[], name='discount'
        ),
        feature
    )
    
    return infer_step

# get reward_fn & action_fn
def _get_rewards(element):
    """Calculates reward for the actions."""

    def _calc_reward(x):
        """Calculates reward for a single action."""
        r0 = lambda: tf.constant(0.0)
        r1 = lambda: tf.constant(1.0)
        r2 = lambda: tf.constant(2.0)
        r3 = lambda: tf.constant(3.0)
        r4 = lambda: tf.constant(4.0)
        r5 = lambda: tf.constant(5.0)
        c1 = tf.equal(x, 1.0)
        c2 = tf.equal(x, 2.0)
        c3 = tf.equal(x, 3.0)
        c4 = tf.equal(x, 4.0)
        c5 = tf.equal(x, 5.0)
        return tf.case(
            [(c1, r1), (c2, r2), (c3, r3),(c4, r4),(c5, r5)], 
            default=r0, exclusive=True
        )

    return tf.map_fn(
        fn=_calc_reward, 
        elems=element['user_rating'], 
        dtype=tf.float32
    )

# ====================================================
# pred app
# ====================================================
app = FastAPI()

# ====================================================
# load trained policy
# ====================================================
trained_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
    os.environ["AIP_STORAGE_URI"], load_specs_from_pbtxt=True
)

# ====================================================
# load vocab
# ====================================================
LOCAL_VOCAB_FILENAME = 'vocab_dict.pkl'

download_blob(
    project_id = data_config.PROJECT_ID,
    bucket_name = data_config.BUCKET_NAME, 
    source_blob_name = 'vocabs/vocab_dict.pkl', 
    destination_file_name= LOCAL_VOCAB_FILENAME
)
filehandler = open(f"{LOCAL_VOCAB_FILENAME}", 'rb')
vocab_dict = pkl.load(filehandler)
filehandler.close()

# ====================================================
# embedding layers
# ====================================================
NUM_OOV_BUCKETS        = 1
GLOBAL_EMBEDDING_SIZE  = 16
MV_EMBEDDING_SIZE      = 32 #32

embs = emb_features.EmbeddingModel(
    vocab_dict = vocab_dict,
    num_oov_buckets = NUM_OOV_BUCKETS,
    global_emb_size = GLOBAL_EMBEDDING_SIZE,
    mv_emb_size = MV_EMBEDDING_SIZE,
)


@app.get(os.environ["AIP_HEALTH_ROUTE"], status_code=200)
def health():
    """
    Handles server health check requests.

    Returns:
       An empty dict.
    """
    return {"status": "healthy"}


@app.post(os.environ["AIP_PREDICT_ROUTE"])
async def predict(request: Request):
    """
    Handles prediction requests.

    Unpacks observations in prediction requests and queries the trained policy for
    predicted actions.

    Args:
    request: Incoming prediction requests that contain observations.

    Returns:
    A dict with the key `predictions` mapping to a list of predicted actions
    corresponding to each observation in the prediction request.
    """
    body = await request.json()
    instances = body["instances"]
    logging.info(f'instances: {instances}') # tmp - debugging

    predictions = []
    
    global_feat_infer = embs._get_global_context_features(instances)
    logging.info(f'global_feat_infer: {global_feat_infer}') # tmp -debugging
    
    arm_feat_infer = embs._get_per_arm_features(instances)
    logging.info(f'arm_feat_infer: {arm_feat_infer}') # tmp -debugging
    
    rewards = _get_rewards(instances)
    logging.info(f'rewards: {rewards}') # tmp -debugging
    
    # this could be replaced with a function that generates random items 
    # or items from a specific list for exploration
    dummy_arm = tf.zeros([1, data_config.PER_ARM_DIM], dtype=tf.float32)
    
    # reshape arm features
    arm_feat_infer = tf.reshape(arm_feat_infer, [data_config.eval_batch_size, data_config.PER_ARM_DIM]) # perarm_dim
    concat_arm = tf.concat([arm_feat_infer, dummy_arm], axis=0)
    
    # flatten global
    flat_global_infer = tf.reshape(global_feat_infer, [data_config.GLOBAL_DIM])
    feature = {'global': flat_global_infer, 'per_arm': concat_arm}
    logging.info(f'feature: {feature}') # tmp -debugging
    
    # get actual reward
    actual_reward = rewards.numpy()[0]
    
    # build trajectory step
    trajectory_step = _get_pred_step(feature, actual_reward)
    
    prediction = trained_policy.action(trajectory_step)

    return {
        "prediction": prediction,
        # "item_id": instances['movie_id'],
    }