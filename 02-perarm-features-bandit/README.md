# Multi-Armed Bandits with per-arm features

> TODO

We **don’t actually have to implement the Environment** in TF-Agents because during training the data is read offline from logs, and during inference, the policy is served in the actual production environment as a SavedModel through Servo (it may be useful to have an environment for running simulation experiments though)

We have to implement specs to create the agent, policies, network etc. 


## (1) build per-arm bandit algorithm locally

> TODO

## (2) Build training application to scale off-policy training with Vertex AI

> TODO

## (3) understand GPU/TPU support and performance profiling for TF-Agents 

> TODO

![alt text](https://github.com/tottenjordan/tf_vertex_agents/blob/main/imgs/agent_profiler_v1.png)


# Why generalize bandits to neural network models?
* NeuralLinear Bandits are linear contextual bandits use the last layer representation of the neural network as the contextual features
* NeuralLinear works well [paper](https://arxiv.org/pdf/1802.09127.pdf)
* Decouples representation learning and uncertainty estimation
* Computationally inexpensive 
* Achieves superior performance on multiple datasets

## What are "per-arm" features?
* In some bandits use cases, each arm has its own features. For example, in movie recommendation problems, the user features play the role of the context and the movies play the role of the arms (aka actions) 
* Each movie has its own features, such as `text description`, `metadata`, `trailer content` features and so on

These problems are often referred to as `arm features problems`

# References
**TF-Agent tutorials:**

(1) [Multi-Armed Bandits with Per-Arm Features](https://www.tensorflow.org/agents/tutorials/per_arm_bandits_tutorial)

* step-by-step guide on how to use the TF-Agents library for contextual bandits problems where the actions (arms) have their own features, such as a list of movies represented by features (genre, year of release, ...)

(2) [Networks](https://www.tensorflow.org/agents/tutorials/8_networks_tutorial)

* define custom networks for your agents
* The networks help us define the model that is trained by agents

(3) [Ranking](https://www.tensorflow.org/agents/tutorials/ranking_tutorial)

* ranking algorithms implemented as part of the TF-Agents Bandits library 
* In a ranking problem, in every iteration an agent is presented with a set of items, and is tasked with ranking some or all of them to a list
* This ranking decision then receives some form of feedback (maybe a user does or does not click on one or more of the selected items for example)
* The goal of the agent is to optimize some metric/reward with the goal of making better decisions over time