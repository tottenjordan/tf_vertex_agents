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
def train_validation(
    project_id: str,
    location: str,
    pipeline_version: str,
    bucket_name: str,
    bq_table_ref: str,
    tf_record_file: str,
    batch_size: int,
    num_actions: int,
    global_dim: int,
    per_arm_dim: int,
    experiment_name: str,
    num_epochs: int = 2,
) -> NamedTuple('Outputs', [
    ('log_dir', str),
]):
    import os
    import time
    import numpy as np
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from google.cloud import aiplatform, storage
    from typing import Dict, List, Any
    
    import tensorflow as tf
    from tf_agents.specs import array_spec
    from tf_agents.specs import tensor_spec
    from tf_agents.policies import policy_saver
    from tf_agents import trajectories
    from tf_agents.trajectories import time_step as ts
    from tf_agents.bandits.policies import policy_utilities
    from tf_agents.bandits.specs import utils as bandit_spec_utils
    from tf_agents.metrics import tf_metrics
    
    # this repo
    from src import train_utils as train_utils
    from src.data_preprocessor import preprocess_utils
    from src.agents import agent_factory as agent_factory
    
    # set experiment config for tracking
    invoke_time       = time.strftime("%Y%m%d-%H%M%S")
    RUN_NAME          = f'run-{invoke_time}'
    BASE_OUTPUT_DIR   = f"gs://{bucket_name}/{experiment_name}/{RUN_NAME}"
    LOG_DIR           = f"{BASE_OUTPUT_DIR}/logs"
    ARTIFACTS_DIR     = f"{BASE_OUTPUT_DIR}/artifacts"
    print(f"BASE_OUTPUT_DIR : {BASE_OUTPUT_DIR}")
    print(f"LOG_DIR         : {LOG_DIR}")
    print(f"ARTIFACTS_DIR   : {ARTIFACTS_DIR}")
    
    aiplatform.init(project=project_id, location=location)
    tensorboard = aiplatform.Tensorboard.create(
        display_name=experiment_name
        , project=project_id
        , location=location
    )
    TB_RESOURCE_NAME = tensorboard.resource_name
    TB_ID = TB_RESOURCE_NAME.split('/')[-1]
    
    # set agent config
    AGENT_TYPE      = 'epsGreedy' # 'LinUCB' | 'LinTS |, 'epsGreedy' | 'NeuralLinUCB'
    AGENT_ALPHA     = 0.1
    EPSILON         = 0.01
    LR              = 0.05
    ENCODING_DIM    = 1
    EPS_PHASE_STEPS = 1000
    GLOBAL_LAYERS   = [global_dim, int(global_dim/2), int(global_dim/4)]
    ARM_LAYERS      = [per_arm_dim, int(per_arm_dim/2), int(per_arm_dim/4)]
    FIRST_COMMON_LAYER = GLOBAL_LAYERS[-1] + ARM_LAYERS[-1]
    COMMON_LAYERS = [
        int(FIRST_COMMON_LAYER),
        int(FIRST_COMMON_LAYER/4)
    ]
    NETWORK_TYPE = "commontower"
    
    # set tensor specs
    observation_spec = {
        'global': tf.TensorSpec([global_dim], tf.float32),
        'per_arm': tf.TensorSpec([num_actions, per_arm_dim], tf.float32) #excluding action dim here
    }
    action_spec = tensor_spec.BoundedTensorSpec(
        shape=[], 
        dtype=tf.int32,
        minimum=tf.constant(0),            
        maximum=num_actions-1, # n degrees of freedom and will dictate the expected mean reward spec shape
        name="action_spec"
    )
    time_step_spec = ts.time_step_spec(observation_spec = observation_spec)

    reward_spec = {
        "reward": array_spec.ArraySpec(
            shape=[batch_size], 
            dtype=np.float32, name="reward"
        )
    }
    reward_tensor_spec = train_utils.from_spec(reward_spec)
    
    # create agent
    global_step = tf.compat.v1.train.get_or_create_global_step()
    agent = agent_factory.PerArmAgentFactory._get_agent(
        agent_type = AGENT_TYPE,
        network_type = NETWORK_TYPE,
        time_step_spec = time_step_spec,
        action_spec = action_spec,
        observation_spec=observation_spec,
        global_layers = GLOBAL_LAYERS,
        arm_layers = ARM_LAYERS,
        common_layers = COMMON_LAYERS,
        agent_alpha = AGENT_ALPHA,
        learning_rate = LR,
        epsilon = EPSILON,
        train_step_counter = global_step,
        output_dim = ENCODING_DIM,
        eps_phase_steps = EPS_PHASE_STEPS,
        summarize_grads_and_vars = False,
        debug_summaries = True
    )
    agent.initialize()
    print(f'agent: {agent.name}')
    
    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        f"{LOG_DIR}", flush_millis=10 * 1000
    )
    train_summary_writer.set_as_default()
    saver = policy_saver.PolicySaver(
        agent.policy, 
        train_step=global_step
    )
    metrics = [
        # tf_metrics.NumberOfEpisodes(),
        # tf_metrics.AverageEpisodeLengthMetric(batch_size=batch_size),
        tf_metrics.AverageReturnMetric(batch_size=batch_size)
    ]
    # create dataset
    raw_dataset = tf.data.TFRecordDataset([tf_record_file])
    parsed_dataset = raw_dataset.map(
        preprocess_utils._parse_record
    ).prefetch(
        tf.data.experimental.AUTOTUNE
    )
    # trajectory function
    def _build_trajectory_from_tfrecord(
        parsed_record: Dict[str, tf.Tensor],
        batch_size: int,
        num_actions: int,
        # policy_info: policies.utils.PolicyInfo
    ) -> trajectories.Trajectory:
        """
        Builds a `trajectories.Trajectory` object from `parsed_record`.

        Args:
          parsed_record: A dict mapping feature names to values as `tf.Tensor`
            objects of type string containing serialized protos.
          policy_info: Policy information specification.

        Returns:
          A `trajectories.Trajectory` object that contains values as de-serialized
          `tf.Tensor` objects from `parsed_record`.
        """
        dummy_rewards = tf.zeros([batch_size, 1, num_actions])

        global_features = tf.expand_dims(
            tf.io.parse_tensor(parsed_record["observation"], out_type=tf.float32),
            axis=1
        )
        observation = {
            bandit_spec_utils.GLOBAL_FEATURE_KEY: global_features
        }

        arm_features = tf.expand_dims(
            tf.io.parse_tensor(parsed_record["chosen_arm_features"], out_type=tf.float32),
            axis=1
        )

        policy_info = policy_utilities.PerArmPolicyInfo(
            chosen_arm_features=arm_features,
            predicted_rewards_mean=dummy_rewards,
            bandit_policy_type=tf.zeros([batch_size, 1, 1], dtype=tf.int32)
        )

        return trajectories.Trajectory(
            step_type=tf.expand_dims(
                tf.io.parse_tensor(parsed_record["step_type"], out_type=tf.int32),
                axis=1
            ),
            observation = observation,
            action=tf.expand_dims(
                tf.io.parse_tensor(parsed_record["action"], out_type=tf.int32),
                axis=1
            ),
            policy_info=policy_info,
            next_step_type=tf.expand_dims(
                tf.io.parse_tensor(
                    parsed_record["next_step_type"], out_type=tf.int32),
                axis=1
            ),
            reward=tf.expand_dims(
                tf.io.parse_tensor(parsed_record["reward"], out_type=tf.float32),
                axis=1
            ),
            discount=tf.expand_dims(
                tf.io.parse_tensor(parsed_record["discount"], out_type=tf.float32),
                axis=1
            )
        )
    
    # train job
    list_o_loss = []
    # Reset the train step
    agent.train_step_counter.assign(0)

    print(f"starting train job...")
    start_time = time.time()
    # tf.profiler.experimental.start(LOG_DIR)
    for i in range(num_epochs):

        print(f"epoch: {i+1}")

        for parsed_record in parsed_dataset:

            _trajectories = _build_trajectory_from_tfrecord(
                parsed_record, batch_size, num_actions
            )

            step = agent.train_step_counter.numpy()
            loss = agent.train(experience=_trajectories)
            list_o_loss.append(loss.loss.numpy())

            train_utils._export_metrics_and_summaries(
                step=i, 
                metrics=metrics
            )

            # print step loss
            if step % 10 == 0:
                print(
                    'step = {0}: train loss = {1}'.format(
                        step, round(loss.loss.numpy(), 2)
                    )
                )
    # tf.profiler.experimental.stop()
    runtime_mins = int((time.time() - start_time) / 60)
    print(f"train runtime_mins: {runtime_mins}")
    
    # one time upload
    aiplatform.upload_tb_log(
        tensorboard_id=TB_ID,
        tensorboard_experiment_name=experiment_name,
        logdir=LOG_DIR,
        experiment_display_name=experiment_name,
        run_name_prefix=RUN_NAME,
        # description=description,
    )
    
    return (
        f"{LOG_DIR}"
    )
