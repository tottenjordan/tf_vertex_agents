# Contextual Bandits for RecSys with Tensorflow Agents and Vertex AI 

> The goal of reinforcement learning (RL) is to build an autonomous agent that takes a sequence of actions to maximize a utility function by interacting with an external, unknown environment

## How to use this repo

This repo is organized across several notebooks. Complete the first two notebooks to setup your workspace environment and prepare the Movielens dataset. From there, you can choose your own adventure and complete any subdirectory, in any order. 

Below are the high-level objectives of each notebook or set of examples. *See READMEs in each subdirectory for more details*

* [00-env-setup.ipynb](00-env-setup.ipynb) - establish naming conventions and env config for repo
* [00-movielens-data-prep.ipynb](00-movielens-data-prep.ipynb) - prepare movielens datasets for retreival and ranking use cases
* [01-online-bandit-simulation/](01-online-bandit-simulation/) - implement custom `environment` for online simulation
* [02-supervised-to-bandit-training/](02-supervised-to-bandit-training/) - train contextual bandits with previously collected (logged) data
* [03-ranking/](03-ranking/) - train contextual bandits for ranking problems
* [04-pipelines/](04-pipelines/) - MLOps for contextual bandits

## Introduction to Contextual Bandit problems

The Contextual Bandit (CB) problem is a special case of Reinforcement Learning: an agent collects rewards in an environment by taking some actions after observing some state of the environment. The main difference between general RL and CB is that in CB, we assume that **the action taken by the agent does not influence the next state of the environment**. Therefore, agents do not model state transitions, credit rewards to past actions, or "plan ahead" to get to reward-rich states.

As in other RL domains, the goal of a Contextual Bandit agent is to find a *policy* that collects as much reward as possible. It would be a mistake, however, to always try to exploit the action that promises the highest reward, because then there is a chance that we miss out on better actions if we do not explore enough. This is the main problem to be solved in CB, often referred to as the *exploration-exploitation dilemma*.

### Example: movie recommendations

Suppose we are looking to improve user engagement by recommending to a user the best movie from a set of movies. 
* The movies are the *candidate arms*
* *Context* could include user features (e.g., location, device, demographics), as well as additional features about each item (movie)
* *Feedback* could be whether the user clicked or watched a movie

<img src='imgs/deep_rl_mab_example_v3.png' width='800' height='545'>

### Problem formulation

basic questions we need to answer:
* how is the **environment defined:** (e.g., real or simulated environmnet)
* how do we **incentivize the agent:** (e.g., linear or non-linear rewards)
* structure and logic of the **policy** (e.g., neuarl networks or look-up tables)
* Choose a **training algorithm**

### Training data used in this repo

We use the
[MovieLens 100K dataset](https://www.kaggle.com/prajitdatta/movielens-100k-dataset)
to build a simulation environment that frames the recommendation problem:

1.  User vectors are the environment observations;
2.  Movie items to recommend are the agent actions applied on the environment;
3.  Approximate user ratings are the environment rewards generated as feedback
    to the observations and actions.

## TF-Agents concepts to understand

See the [tf_agents/bandits](https://github.com/tensorflow/agents/blob/master/tf_agents/bandits) repository for ready-to-use bandit environments, policies, and agents 

### Environments
In TF-Agents, the environment class serves the role of giving information on the current state (this is called **observation** or **context**), receiving an action as input, performing a state transition, and outputting a reward. This class also takes care of resetting when an episode ends, so that a new episode can start. This is realized by calling a `reset` function when a state is labelled as "last" of the episode. For more details, see the [TF-Agents environments tutorial](https://github.com/tensorflow/agents/blob/master/docs/tutorials/2_environments_tutorial.ipynb).

As mentioned above, CB differs from general RL in that actions do not influence the next observation. Another difference is that in Bandits, there are no "episodes": every time step starts with a new observation, independently of previous time steps.


### Policies
In Reinforcement Learning terminology, policies map an observation from the environment to an action or a distribution over actions. In TF-Agents, observations from the environment are contained in a named tuple `TimeStep('step_type', 'discount', 'reward', 'observation')`, and policies map timesteps to actions or distributions over actions. Most policies use `timestep.observation`, some policies use `timestep.step_type` (e.g. to reset the state at the beginning of an episode in stateful policies), but `timestep.discount` and `timestep.reward` are usually ignored. 

Policies are related to other components in TF-Agents in the following way:
* Most policies have a neural network to compute actions and/or distributions over actions from `TimeSteps` 
* Agents can contain one or more policies for different purposes, e.g. a main policy that is being trained for deployment, and a noisy policy for data collection
* Policies can be saved/restored, and can be used indepedently of the agent for data collection, evaluation etc.

For more details, see the [TF-Agents Policy tutorial](https://github.com/tensorflow/agents/blob/master/docs/tutorials/3_policies_tutorial.ipynb).


### Agents
Agents take care of changing the policy based on training samples (represented as trajectories)

> TODO

### Trajectorties
In TF-Agents, `trajectories` are the training examples used to train an agent. More specifically, they are named tuples that contain samples from previous steps taken. These samples are then used by the agent to train and update the policy. In RL, trajectories must contain information about the current state, the next state, and whether the current episode has ended

> TODO: example trajectory

The [functools](https://docs.python.org/3/library/functools.html) package is commonly used to develop RL agents with TF-Agents because it provides a number of useful functions for working with nested structures of tensors 
* These functions can be used to create and manipulate data structures such as arrays, lists, and dictionaries, which are commonly used in RL algorithms
* Also used for higher-order functions (i.e., functions that act on or return other functions). In general, any callable object can be treated as a function for the purposes of this module


### Regret
Bandits' most important metric is *regret*, calculated as the difference between the reward collected by the agent and the expected reward of an oracle policy that has access to the reward functions of the environment. The [RegretMetric](https://github.com/tensorflow/agents/blob/master/tf_agents/bandits/metrics/tf_metrics.py) thus needs a `baseline_reward_fn` function that calculates the best achievable expected reward given an observation


### Training Agents

> TODO

#### Track Agent training performance with Vertex AI Experiments

> Vertex AI enables users to track the steps (e.g., preprocessing, training) of an experiment run, and track inputs (e.g., algorithm, parameters, datasets) and outputs (fe.g.,, models, checkpoints, metrics) of those steps.

**How are parameters and metrics stored / organized?**
* `Experiments` describe a context that groups your runs and the artifacts you create into a logical session. For example, in this notebook you create an Experiment and log data to that experiment.
* an `Experiment Run` represents a single path/avenue that you executed while performing an experiment. A run includes artifacts that you used as inputs or outputs, and parameters that you used in this execution. An Experiment can contain multiple runs.

**Loggin' and sharin'**
* Use the Vertex AI (Python) SDK to track metrics, params, and metadata for models trained (either local or cloud) for each experiment across several experiment runs
* Link a [Managed TensorBoard](https://cloud.google.com/vertex-ai/docs/experiments/tensorboard-introduction) instance to visualize, compare, and share ML experiments with your team 
* Save time and effort by [Autologging experiment data](https://cloud.google.com/vertex-ai/docs/experiments/autolog-data) from training jobs, and [pipeline runs](https://cloud.google.com/vertex-ai/docs/experiments/add-pipelinerun-experiment) 
* To learn more, see [Introduction to Vertex AI Experiments](https://cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments)

### Profiling Agents

> TODO

![alt text](https://github.com/tottenjordan/tf_vertex_agents/blob/main/imgs/getting_profiler.png)

#### Input pipeline bottleneck analysis

> TODO

![alt text](https://github.com/tottenjordan/tf_vertex_agents/blob/main/imgs/tb_input_bottleneck_analysis.png)

# Repo details

> **TODO**

> 00 - Helper notebooks for staying organized, minimizing overhead, and validating reproducibility
* [00-env-setup.ipynb](00-env-setup.ipynb)
* [00-movielens-data-prep.ipynb](00-movielens-data-prep.ipynb)

> 01 - Simulate a real-world interaction environment of users and their respective preferences.
* [01-online-bandit-simulation/](01-online-bandit-simulation/)
  * [01a-train-bandit-mf-env-simulation.ipynb](01-online-bandit-simulation/01a-train-bandit-mf-env-simulation.ipynb)
  * [01b-build-training-image.ipynb](01-online-bandit-simulation/01b-build-training-image.ipynb)
  * [01c-scale-bandit-simulation-vertex.ipynb](01-online-bandit-simulation/01c-scale-bandit-simulation-vertex.ipynb)
  
> 02 - Convert a training dataset fit for supervised learning (i.e., a labled dataset with logged user feedback) and use it to train a contextual bandit algorithm, where the labels (user feedback) is used as a proxy for immediate rewards
* [02-supervised-to-bandit-training/](02-supervised-to-bandit-training/)
  * [02a-train-supervised-bandit-simulation.ipynb](02-supervised-to-bandit-training/02a-train-supervised-bandit-simulation.ipynb)
  * [02b-train-supervised-bandit-locally.ipynb](02-supervised-to-bandit-training/02b-train-supervised-bandit-locally.ipynb)
  * [02c-accelerated-bandits.ipynb](02-supervised-to-bandit-training/02c-accelerated-bandits.ipynb)
  * [02d-build-training-image.ipynb](02-supervised-to-bandit-training/02d-build-training-image.ipynb)
  * [02e-scale-bandit-training-vertex.ipynb](02-supervised-to-bandit-training/02e-scale-bandit-training-vertex.ipynb)
  * [02f-cpr-deploy-bandit-policy.ipynb](02-supervised-to-bandit-training/02f-cpr-deploy-bandit-policy.ipynb)
  
> 03 - (WIP) Train ranking policies that score set of items; currently explores `score_vector` and `cascading_feedback` frameworks
* [03-ranking/](03-ranking/)
  * [03a-ranking-bandit-local-train.ipynb](03-ranking/03a-ranking-bandit-local-train.ipynb)
  * [03x-baseline-train-ranking-bandit.ipynb](03-ranking/03x-baseline-train-ranking-bandit.ipynb)
  
> 04 - (WIP) Step-by-step guide to MLOps tasks --> then how to orchestrate them with pipelines
* [04-pipelines/](04-pipelines/)
  * [01 Step-by-Step MLOps Demo](04-pipelines/01-step_by_step_sdk_tf_agents_bandit/)
    * [README](04-pipelines/01-step_by_step_sdk_tf_agents_bandit/README.md)
    * [notebook](04-pipelines/01-step_by_step_sdk_tf_agents_bandit/01-train_deploy_mab.ipynb)
    * [src/](04-pipelines/01-step_by_step_sdk_tf_agents_bandit/src/)
  * [02 End-to-End MLOps Pipeline Demo](04-pipelines/02-mlops_pipeline_tf_agents_bandit/)
    * [README](04-pipelines/02-mlops_pipeline_tf_agents_bandit/README.md)
    * [notebook](04-pipelines/02-mlops_pipeline_tf_agents_bandit/02-mab_mlops_pipeline.ipynb)
    * [src/](04-pipelines/02-mlops_pipeline_tf_agents_bandit/src/)
    
> (WIP) Discussion and commentary for related RL and RecSys topics
* [learning/](learning/)
  * [RL Agents used in this repo](learning/agents.md)
  * [RL for RecSys -> concepts and practical considerations](learning/rl-for-recsys.md)
  * [Implementing RL with TF-Agents](learning/tf-agents-overview.md)
  * TODO

