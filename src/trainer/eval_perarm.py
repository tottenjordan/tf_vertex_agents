"""Run eval loop for bandit per-arm feature Agents"""
import os
import numpy as np

# tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from tf_agents.bandits.policies import policy_utilities

# this project
from src.data import data_config as data_config
from src.networks import encoding_network as emb_features
from src.utils import reward_factory, train_utils

# ====================================================
# run bandit eval
# ====================================================
def _run_bandit_eval(
    policy,
    data,
    eval_batch_size: int,
    per_arm_dim: int,
    global_dim: int,
    vocab_dict: dict,
    num_oov_buckets: int,
    global_emb_size: int,
    mv_emb_size: int,
):
    actual_rewards = []
    predicted_rewards = []
    trouble_list = []
    train_loss_results = []
    
    embs = emb_features.EmbeddingModel(
        vocab_dict = vocab_dict,
        num_oov_buckets = num_oov_buckets,
        global_emb_size = global_emb_size,
        mv_emb_size = mv_emb_size,
        max_genre_length = data_config.MAX_GENRE_LENGTH
    )
    
    dummy_arm = tf.zeros([eval_batch_size, per_arm_dim], dtype=tf.float32)

    for x in data:
        
        filter_mask = None
        # get feature tensors

        global_feat_infer = embs._get_global_context_features(x)
        arm_feat_infer = embs._get_per_arm_features(x)

        rewards = reward_factory._get_rewards(x)

        # reshape arm features
        arm_feat_infer = tf.reshape(arm_feat_infer, [eval_batch_size, per_arm_dim])
        concat_arm = tf.concat([arm_feat_infer, dummy_arm], axis=0)

        # flatten global
        flat_global_infer = tf.reshape(global_feat_infer, [global_dim])
        feature = {'global': flat_global_infer, 'per_arm': concat_arm}

        # get actual reward
        actual_reward = rewards.numpy()[0]

        # build trajectory step
        trajectory_step = train_utils._get_eval_step(feature, actual_reward)

        # pred w/ trained agent
        prediction = policy.action(trajectory_step)

        predicted_rewards_mean = prediction.info.predicted_rewards_mean #[0]
        # pred_rewards_mean_list.append(predicted_rewards_mean)

        predicted_reward_tf = tf.gather(
            predicted_rewards_mean,
            prediction.action, 
            batch_dims=0, 
            axis=-1
        )
        # pred_reward = float(round(predicted_reward_tf.numpy()))
        pred_reward = predicted_reward_tf.numpy()

        filter_mask = tf.equal(
            tf.squeeze(prediction.info.bandit_policy_type),
            policy_utilities.BanditPolicyType.GREEDY
        )

        if filter_mask is None:
            trouble_list.append(pred_reward)
        else:
            pred_loss = tf.keras.metrics.mean_squared_error(
                rewards, predicted_reward_tf
            )
            train_loss_results.append(pred_loss)
            # log the predicted rewards
            predicted_rewards.append(pred_reward)
            # log the actual reward
            actual_rewards.append(actual_reward)
            
        # When the uniform random policy is used, the loss is meaningless for evaluation
        # > discard preds from uniform random policy
        # > keep preds from greedy policy, 
        # TODO: replace hard-coded values with parameterized filter/mask
#         if pred_reward < 0:
#             trouble_list.append(pred_reward)
#         elif pred_reward > 5:
#             trouble_list.append(pred_reward)
#         else:
#             predicted_rewards.append(pred_reward)

#             pred_loss = tf.keras.metrics.mean_squared_error(
#                 rewards, predicted_reward_tf
#             )
#             train_loss_results.append(pred_loss)
#             logged_rewards.append(actual_reward)

    # calculate avg loss
    avg_eval_loss = tf.reduce_mean(train_loss_results)

    return (
        avg_eval_loss,
        predicted_rewards,
        actual_rewards,
    )