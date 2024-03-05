"""
A network that operates on repeated global and arm features
"""

# REPATED_NETWORK - for sequential data processing
"""
This network operates on observations with single and repeated features.

  The below depicted observation has two main branches: a global and a repeated.
  Both of these have a single (non-repeated) feature spec and a list of repeated
  feature specs.

  When called, the network sends all the repeated features through their own
  subnetworks (a few hidden layers), then reduces the output so the repetition
  dimensions are removed. After reduction, all outputs are concatenated together
  with the single feature. After another subnet of hidden layers, the global and
  the per-arm branches are concatenated and led through a final common subnet.

  For the design of this network see go/sc-repeated-network-design.
  
  Output of this small tower, the reduction function acts to reduce all features to:
      
      [batch_size, feature_dim] for global features and 
      [batch_size, num_actions, feature_dim] for per-arm features

  From this point the network is the same as the original per-arm network.
  
The observations must follow the observation spec as follows:

"""
obs_spec = {
    'global': {
        'single': TensorSpec(shape=[global_single_dim]),
        'repeated': [
            TensorSpec(shape=[num_repetitionsS1E1,global_repeated_dim1]),
            TensorSpec(
                shape=[
                    global_num_repetitionsS2E1,
                    global_num_repetitionsS2E2,
                    global_repeated_dim2
                ]
            ),
        ]
    },
    'per_arm': {
        'single': TensorSpec(shape=[num_arms, arm_single_dim]),
        'repeated': [
            TensorSpec(
                shape=[
                    num_arms,arm_num_repetitionsS1E1,
                    arm_num_repetitionsS1E2,
                    arm_repeated_dim1
                ]
            ),
            TensorSpec(
                shape=[
                    num_arms,
                    arm_num_repetitionsS2E1,
                    arm_repeated_dim2
                ]
            ),
        ]
    }
}
