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