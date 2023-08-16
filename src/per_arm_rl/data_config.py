# ================================
# Agents
# ================================
# AGENT_TYPE      = 'epsGreedy' # 'LinUCB' | 'LinTS |, 'epsGreedy' | 'NeuralLinUCB'

# Parameters for linear agents (LinUCB and LinTS).
AGENT_ALPHA     = 0.1

# Parameters for neural agents (NeuralEpsGreedy and NerualLinUCB).
EPSILON         = 0.01
LR              = 0.05

# Parameters for NeuralLinUCB
ENCODING_DIM    = 1
EPS_PHASE_STEPS = 1000

# ================================
# Agent's Preprocess Network
# ================================
# NETWORK_TYPE    = "dotproduct" # 'commontower' | 'dotproduct'

# if AGENT_TYPE == 'NeuralLinUCB':
#     NETWORK_TYPE = 'commontower'
    
GLOBAL_LAYERS   = [64, 32]
ARM_LAYERS      = [64, 32]
COMMON_LAYERS   = [32, 16]

# TODO - parametrize
NUM_OOV_BUCKETS        = 1
GLOBAL_EMBEDDING_SIZE  = 16
MV_EMBEDDING_SIZE      = 32 #32

