# Bandit Rankers

> **TODO**

## Overview

* The contextual bandits approach is classified as an extension of multi-armed bandits
* a contextual multi-armed bandit problem is a simplified reinforcement learning algorithm where the agent takes an action from a set of possible actions 

### The **Bandit Ranking** agent will be similar to the `NeuralEpsilonGreedy` agent. 

Main differences:
* The item features are stored in the `per_arm` part of the observation, in the order of how they are recommended
* Since this ordered list of items expresses what action was taken by the policy,
the `action` value of the trajectory is not used by the agent.

> Note: difference between the "per-arm" observation recieved by the policy vs the agent:

> * While the agent receives the items in the recommendation slots, the policy receives the items that are available for recommendation. 
> * The user is responsible for converting the observation to the syntax required by the agent.


### Training data structure

The training observation contains the global features and the features of the items in the recommendation slots 
* The item features are stored in the `per_arm` part of the observation, in the order of how they are recommended
* Note: since this ordered list of items expresses what action was taken by the policy, the action value of the trajectory is not used by the agent

## The online learning paradigm
* RL deployments that run batch training and push models at a specific cadence are commonly categorized into `off-policy learning`. And, they are prone to `system bias` because of the long delay from user feedback to model updates
> * In RL terms, this means the algorithm accumulates *regret* as user’s preferences change
* Supervised learning approaches also ignore *exploration*, which is crucial to respond to changing items and user preferences. 
* However, an online approach using continuous training can minimize overall regret
> * match users and items with real-time algorhtms and systems
> * Online and on-policy learning with principled exploration (aka `bandits`)
> * In practice, on-policy systems may reduce the end-to-end policy update delay to < 1 hour, where their batch counterparts could have a delay of several hours to several days 

### Online challenges 
* **Large output space** --> large exploration space without action space reduction
* **Efficient bandits learning in real-time** --> modeling user and context with a *good* tradeoff between accuracy and learning efficiency

#### Online + Offline
One approach is to use concepts from both online and offline:
* Offline: dual-encoder models for learning user and item embeddings
* Online: sparse [bipartite graph](https://www.geeksforgeeks.org/bipartite-graph/#:~:text=A%20Bipartite%20Graph%20is%20a,V%20and%20v%20to%20U.) created from offline embeddings

This can help with (a) the cold-start problem by connecting users with fresh content, as well as both (b) corpus exploration and (c) interest exploration  

## Understanding model variables and gradients during training with TensorBoard

### Historgrams

![alt text](https://github.com/tottenjordan/tf_vertex_agents/blob/main/imgs/tb_histo_grams_full.png)

### Distributions

![alt text](https://github.com/tottenjordan/tf_vertex_agents/blob/main/imgs/distributions_ranking.png)

### Literature
1. [Cascading Linear Submodular Bandits: Accounting for Position Bias and Diversity in Online Learning to Rank](http://auai.org/uai2019/proceedings/papers/248.pdf), G. Hiranandani, H. Singh, P. Gupta, I. A. Burhanuddin, Z. Wen and B. Kveton, 35th Conference on Uncertainty in Artificial Intelligence (2019)
> * account for both position bias and diversity in forming the list of items to recommend
2. [Contextual Combinatorial Cascading Bandits](http://proceedings.mlr.press/v48/lif16.html), , S. Li, B. Wang, S. Zhang, W. Chen, Proceedings of The 33rd International Conference on Machine Learning, PMLR 48:1245-1253, 2016