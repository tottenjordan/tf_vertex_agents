# Bandit Rankers

> **TODO**

## Objectives
  * `baseline-ranking-agents.ipynb` - build baseline ranking bandits: score vector and cascading feedback
  * `TODO` - build ranking image
  * `TODO` - submit ranking train job to Vertex AI
  * `TODO` - serving ranking bandit with Vertex AI

## Overview

* The contextual bandits approach is classified as an extension of multi-armed bandits
* a contextual multi-armed bandit problem is a simplified reinforcement learning algorithm where the agent takes an action from a set of possible actions 

### The **Bandit Ranking** agent will be similar to the `NeuralEpsilonGreedy` agent. 

Main differences:
* The item features are stored in the `per_arm` part of the observation, in the order of how they are recommended
* Since this ordered list of items expresses what action was taken by the policy,
the `action` value of the trajectory is not used by the agent.

*Note: difference between the "per-arm" observation recieved by the policy vs the agent:*

> * While the agent receives the items in the recommendation slots, the policy receives the items that are available for recommendation. 
> * The user is responsible for converting the observation to the syntax required by the agent.


### Training data structure

The training observation contains the global features and the features of the items in the recommendation slots 
* The item features are stored in the `per_arm` part of the observation, in the order of how they are recommended
* Note: since this ordered list of items expresses what action was taken by the policy, the action value of the trajectory is not used by the agent

> **TODO**

## Understanding model variables and gradients during training with TensorBoard


### Historgrams

> **TODO**

![alt text](https://github.com/tottenjordan/tf_vertex_agents/blob/main/imgs/tb_histo_grams_full.png)


### Distributions

> **TODO**

![alt text](https://github.com/tottenjordan/tf_vertex_agents/blob/main/imgs/distributions_ranking.png)

### References
1. [Cascading Linear Submodular Bandits: Accounting for Position Bias and Diversity in Online Learning to Rank](http://auai.org/uai2019/proceedings/papers/248.pdf), G. Hiranandani, H. Singh, P. Gupta, I. A. Burhanuddin, Z. Wen and B. Kveton, 35th Conference on Uncertainty in Artificial Intelligence (2019)
> * account for both position bias and diversity in forming the list of items to recommend
2. [Contextual Combinatorial Cascading Bandits](http://proceedings.mlr.press/v48/lif16.html), , S. Li, B. Wang, S. Zhang, W. Chen, Proceedings of The 33rd International Conference on Machine Learning, PMLR 48:1245-1253, 2016
