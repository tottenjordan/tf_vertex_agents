# Supervised Contextual Bandits with per-arm features

> in this section, we use a training dataset fit for supervised learning (i.e., a labled dataset with logged user feedback) and use it to train a contextual bandit algorithm, where the labels (user feedback) is used as a proxy for immediate rewards

## Objectives

  * `02a-train-supervised-bandit-locally.ipynb` - Use labeled dataset to train bandit locally (same as 1 except no "environment")
  * `02c-accelerated-bandits.ipynb` - (Optional) profile train job on local notebook GPU
  * `02d-build-training-image.ipynb` - build docker image for scaling training with Vertex AI
  * `02e-scale-bandit-training-vertex.ipynb` - submit train job to Vertex AI
  * `02f-cpr-deploy-bandit-policy.ipynb` - deploy trained bandit policy to Vertex AI online endpoint

## Supervised-to-bandit datasets

> TODO

## Contextual Bandits with per-arm features
In some Bandits use cases, each arm has its own features. For movie recommendations, the movies represent arms (aka `actions`) an agent can choose from. The features describing each movie (or `arm`) are referred to as `per-arm` features. The `per-arm` features for movies could be `title`, `text description`, `genre`, `starring actors`, etc. The ability to process these additional arm features has shown to be effective in learning complex relationships e.g., user preferences evolving over time

These problems are often referred to as `per-arm features problems`

> TODO: include visual illustrating difference between (1) multi-armed bandit and (2) contextual bandit with per-arm features

A **naive implementation** could formulate the problem by having user information as the `global context` and each arm as `movie_1`, `movie_2`, ..., `movie_K`, but this approach has multiple shortcomings:

* The number of actions would have to be all the movies in the system and it is cumbersome to add a new movie.
* The agent has to learn a model for every single movie.
* Similarity between movies is not taken into account.

Instead of numbering the movies, we can do something more intuitive: we can represent movies with a set of features (e.g., `title`, `text description`, `genre`, etc.). The advantages of this approach include:

* Generalisation across movies.
* The agent learns just one reward function that models reward with user and movie features.
* Easy to remove from, or introduce new movies to the system

In this new setting, the number of actions does not even have to be the same in every time step.
  
### Per-arm features in TF-Agents
* Many agents implemented in `tf-agents` can run on environments that have features for its actions. These environments are feferred to as `per-arm environments`
* Reward is modeled not per-arm, but globally

Implementing bandits with per-arm features in TF-Agents requires special attention to the following components:
* Observation spec and observations,
* Action spec and actions,
* Implementation of specific policies and agents.

### Where to capture arm features
In RL, the agent receives an `observation` at every time step and chooses an action. The action is applied to the environment and the environment returns a reward and a new `observation`. The agent trains a policy to choose actions to maximize the sum of rewards, also known as return.

The `observation_spec` and the `action_spec` methods return a nest of `(Bounded)ArraySpecs` that describe the name, shape, datatype and ranges of the observations and actions respectively:

```python
observation_spec = {
    'global': tf.TensorSpec([GLOBAL_DIM], tf.float32),
    'per_arm': tf.TensorSpec([NUM_ACTIONS, PER_ARM_DIM], tf.float32) #excluding action dim here
}
```

the action spec
```python
action_spec = tensor_spec.BoundedTensorSpec(
    shape=[], 
    dtype=tf.int32,
    minimum=tf.constant(0),            
    maximum=NUM_ACTIONS-1, # n degrees of freedom and will dictate the expected mean reward spec shape
    name="action_spec"
)
```
### GreedyRewardPredictionAgent with Per-Arm features

> TODO: add diagram of per-arm feature neural network for eps-Greedy Agent


## Notes on implementation

We **don’t actually have to implement the Environment** in TF-Agents because during training, the data is read offline from logs, and during inference, the policy is served in the actual production environment as a `SavedModel` (it may be useful to have an environment for running simulation experiments though)

We implement `specs` to create the agent, policies, network etc. 

### TensorSpecs

> TODO

### Training data structure

> TODO

## (1) build per-arm bandit algorithm locally

> TODO

## (2) Build training application to scale off-policy training with Vertex AI

> TODO

## (3) understand GPU/TPU support and performance profiling for TF-Agents 

> TODO

![alt text](https://github.com/tottenjordan/tf_vertex_agents/blob/main/imgs/getting_profiler.png)


# Why generalize bandits to neural network models?
* NeuralLinear Bandits are linear contextual bandits use the last layer representation of the neural network as the contextual features
* NeuralLinear works well [paper](https://arxiv.org/pdf/1802.09127.pdf)
* Decouples representation learning and uncertainty estimation
* Computationally inexpensive 
* Achieves superior performance on multiple datasets

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