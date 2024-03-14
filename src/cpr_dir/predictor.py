import os
import sys
import logging
import numpy as np
import pickle as pkl
from typing import Dict, Any, Tuple

logging.disable(logging.WARNING)

# google cloud
from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils
from google.cloud import storage

# tensorflow
import tensorflow as tf
import tf_agents
from tf_agents.policies import py_tf_eager_policy
from tf_agents.trajectories import time_step as ts

# this repo
sys.path.extend([f'./{name}' for name in os.listdir(".") if os.path.isdir(name)])

from user_code import pred_config as pred_config
from user_code import emb_features_pred as emb_features
from user_code import reward_factory as reward_factory

os.environ["PROJECT_ID"] = pred_config.PROJECT_ID

# ==================================
# get trajectory step for prediction
# ==================================
def _get_pred_step(feature, reward_np):
    
    infer_step = ts.TimeStep(
        tf.constant(ts.StepType.FIRST, dtype=tf.int32, shape=[],name='step_type'),
        tf.constant(reward_np, dtype=tf.float32, shape=[], name='reward'),
        tf.constant(1.0, dtype=tf.float32, shape=[], name='discount'),
        feature
    )
    
    return infer_step

# ==================================
# prediction logic
# ==================================
class BanditPolicyPredictor(Predictor):
    
    """
    Interface of the Predictor class for Custom Prediction Routines.
    
    The Predictor is responsible for the ML logic for processing a prediction request.
    
    Specifically, the Predictor must define:
        (1) How to load all model artifacts used during prediction into memory.
        (2) The logic that should be executed at predict time.
    
    When using the default PredictionHandler, the Predictor will be invoked as follows:
    
      predictor.postprocess(predictor.predict(predictor.preprocess(prediction_input)))
    
    """
    
    def __init__(self):
        
        self._local_vocab_filename = "./vocab_dict.pkl"
        self._num_oov_buckets = pred_config.NUM_OOV_BUCKETS
        self._global_embedding_size = pred_config.GLOBAL_EMBEDDING_SIZE
        self._mv_embedding_size = pred_config.MV_EMBEDDING_SIZE
        return
        
    def load(self, artifacts_uri: str):
        """
        Loads trained policy dir & vocabulary
        Args:
            artifacts_uri (str):
                Required. The value of the environment variable AIP_STORAGE_URI.
                has `artifacts/` as a sub directory 
        
        """
        prediction_utils.download_model_artifacts(artifacts_uri)
        
        # init deploy policy
        self._deployment_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
            'artifacts', load_specs_from_pbtxt=True
        )
        
        # load vocab dict
        filehandler = open(f"{self._local_vocab_filename}", 'rb')
        self._vocab_dict = pkl.load(filehandler)
        filehandler.close()
        
        # only if no custom preprocessor is defined
        # self._preprocessor = preprocessor
        
    def preprocess(self, prediction_input: Dict): # -> Tuple[Dict, float]:
        """
        Args:
            prediction_input (Any):
                Required. The prediction input that needs to be preprocessed.
        Returns:
            The preprocessed prediction input.        
        """
        # inputs = super().preprocess(prediction_input)
        
        dummy_arm = tf.zeros([1, pred_config.PER_ARM_DIM], dtype=tf.float32)
        
        batch_size = len(prediction_input) #["instances"])
        assert batch_size == 1, 'prediction batch_size must be == 1'
        
        self._embs = emb_features.EmbeddingModel(
            vocab_dict = self._vocab_dict,
            num_oov_buckets = self._num_oov_buckets,
            global_emb_size = self._global_embedding_size,
            mv_emb_size = self._mv_embedding_size,
        )
        
        # preprocess example
        rebuild_ex = {}

        for x in prediction_input: #["instances"]:
            rebuild_ex['target_movie_id'] = tf.constant([x["target_movie_id"]], dtype=tf.string)
            rebuild_ex['target_movie_rating'] = tf.constant([x["target_movie_rating"]], dtype=tf.float32)
            rebuild_ex['target_rating_timestamp'] = tf.constant([x["target_rating_timestamp"]], dtype=tf.int64)
            rebuild_ex['target_movie_genres'] = tf.constant([x["target_movie_genres"]], dtype=tf.string)
            rebuild_ex['target_movie_year'] = tf.constant([x["target_movie_year"]], dtype=tf.int64)
            rebuild_ex['target_movie_title'] = tf.constant([x["target_movie_title"]], dtype=tf.string)
            rebuild_ex['user_id'] = tf.constant([x["user_id"]], dtype=tf.string)
            rebuild_ex['user_gender'] = tf.constant([x["user_gender"]], dtype=tf.string)
            rebuild_ex['user_age'] = tf.constant([x["user_age"]], dtype=tf.int64)
            rebuild_ex['user_occupation_text'] = tf.constant([x["user_occupation_text"]], dtype=tf.string)
            rebuild_ex['user_zip_code'] = tf.constant([x["user_zip_code"]], dtype=tf.string)
        
        global_feat_infer = self._embs._get_global_context_features(rebuild_ex)
        logging.info(f'global_feat_infer: {global_feat_infer}')          # tmp - debugging
        
        arm_feat_infer = self._embs._get_per_arm_features(rebuild_ex)    # tmp - debugging
        logging.info(f'arm_feat_infer: {arm_feat_infer}')
    
        rewards = reward_factory._get_rewards(rebuild_ex)
        logging.info(f'rewards: {rewards}')                              # tmp - debugging
        
        actual_reward = rewards.numpy()[0]
        logging.info(f'actual_reward: {actual_reward}')                  # tmp - debugging
        
        arm_feat_infer = tf.reshape(arm_feat_infer, [1, pred_config.PER_ARM_DIM])
        concat_arm = tf.concat([arm_feat_infer, dummy_arm], axis=0)      # tmp - debugging
        
        # flatten global
        flat_global_infer = tf.reshape(global_feat_infer, [pred_config.GLOBAL_DIM])
        feature = {'global': flat_global_infer, 'per_arm': concat_arm}
        logging.info(f'feature: {feature}')                              # tmp - debugging
        
        trajectory_step = _get_pred_step(feature, actual_reward)
        logging.info(f'trajectory_step: {trajectory_step}')
        
        # prediction = self._deployment_policy.action(trajectory_step)
        
        return trajectory_step
    
    def predict(self, instances) -> Dict:
        """
        Performs prediction i.e., policy takes action
        """
        # prediction = self._deployment_policy.action(instances) # trajectory_step
        # return {"predictions": prediction}
        return self._deployment_policy.action(instances)
        

    def postprocess(self, prediction_results: Any) -> Any:
        """ 
        Postprocesses the prediction results
        
        TODO:
             Convert predictions to item IDs
             
        """
        processed_pred_dict = {
            "bandit_policy_type" : int(prediction_results.info.bandit_policy_type[0]),
            "chosen_arm_features" : prediction_results.info.chosen_arm_features.tolist(),
            "predicted_rewards_mean" : prediction_results.info.predicted_rewards_mean.tolist(),
            "action" : int(prediction_results.action.tolist()),
        }
        
        return processed_pred_dict
