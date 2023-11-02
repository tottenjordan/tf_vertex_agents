# Vertex Agents & Bandits

> Contextual Bandits for RecSys with TF Agents and Vertex AI 

## How to use this repo

This repo is organized across several notebooks. Complete the first two notebooks to setup your workspace environment and prepare the Movielens dataset. From there, you can choose your own adventure and complete any subdirectory, in any order. 

Below are the high-level objectives of each notebook or set of examples. *See READMEs in each subdirectory for more details*

* [00-env-setup.ipynb](00-env-setup.ipynb) - establish naming conventions and env config for repo
* [00-movielens-data-prep.ipynb](00-movielens-data-prep.ipynb) - prepare movielens datasets to model retreival and ranking use cases
* [01-baseline-perarm-bandit/](01-baseline-perarm-bandit/) - implement custom `environment` for training multi-armed bandits
* [02-perarm-features-bandit/](02-perarm-features-bandit/) - train contextual bandits with *per-arm* features
* [03-ranking/](03-ranking/) - train contextual bandits for ranking problems
* [04-pipelines/](04-pipelines/) - implement e2e mlops pipelines for multi-armed bandits


## TF-Agents concepts to understand

### Intro: Multi-Armed Bandit problems in TF-Agents
The Multi-Armed Bandit problem (MAB) is a special case of Reinforcement Learning: an agent collects rewards in an environment by taking some actions after observing some state of the environment. The main difference between general RL and MAB is that in MAB, we assume that **the action taken by the agent does not influence the next state of the environment**. Therefore, agents do not model state transitions, credit rewards to past actions, or "plan ahead" to get to reward-rich states.

As in other RL domains, the goal of a MAB agent is to find a *policy* that collects as much reward as possible. It would be a mistake, however, to always try to exploit the action that promises the highest reward, because then there is a chance that we miss out on better actions if we do not explore enough. This is the main problem to be solved in MAB, often referred to as the *exploration-exploitation dilemma*.


![alt text](https://github.com/tottenjordan/tf_vertex_agents/blob/main/imgs/deep_rl_mab_example_v2.png)


See the [tf_agents/bandits](https://github.com/tensorflow/agents/blob/master/tf_agents/bandits) repository for ready-to-use bandit environments, policies, and agents 

### Training data

> **TODO**

We use the
[MovieLens 100K dataset](https://www.kaggle.com/prajitdatta/movielens-100k-dataset)
to build a simulation environment that frames the recommendation problem:

1.  User vectors are the environment observations;
2.  Movie items to recommend are the agent actions applied on the environment;
3.  Approximate user ratings are the environment rewards generated as feedback
    to the observations and actions.


### Environments
In TF-Agents, the environment class serves the role of giving information on the current state (this is called **observation** or **context**), receiving an action as input, performing a state transition, and outputting a reward. This class also takes care of resetting when an episode ends, so that a new episode can start. This is realized by calling a `reset` function when a state is labelled as "last" of the episode. For more details, see the [TF-Agents environments tutorial](https://github.com/tensorflow/agents/blob/master/docs/tutorials/2_environments_tutorial.ipynb).

As mentioned above, MAB differs from general RL in that actions do not influence the next observation. Another difference is that in Bandits, there are no "episodes": every time step starts with a new observation, independently of previous time steps.


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


### Regret
Bandits' most important metric is *regret*, calculated as the difference between the reward collected by the agent and the expected reward of an oracle policy that has access to the reward functions of the environment. The [RegretMetric](https://github.com/tensorflow/agents/blob/master/tf_agents/bandits/metrics/tf_metrics.py) thus needs a `baseline_reward_fn` function that calculates the best achievable expected reward given an observation


### Training Agents

> TODO


### Profiling Agents

> TODO

![alt text](https://github.com/tottenjordan/tf_vertex_agents/blob/main/imgs/getting_profiler.png)


#### Input pipeline bottleneck analysis

> TODO

![alt text](https://github.com/tottenjordan/tf_vertex_agents/blob/main/imgs/tb_input_bottleneck_analysis.png)


# More repo details

> **TODO**

* [00-env-setup.ipynb](00-env-setup.ipynb)
* [01-movielens-data-prep.ipynb](01-movielens-data-prep.ipynb)
* [01-baseline-perarm-bandit/](01-baseline-perarm-bandit/)
  * [01a-build_perarm_tf_agents_model.ipynb](01-baseline-perarm-bandit/01a-build_perarm_tf_agents_model.ipynb)
  * [01b-train_perarm_tf_agents_vertex.ipynb](01-baseline-perarm-bandit/01b-train_perarm_tf_agents_vertex.ipynb)
* [02-perarm-features-bandit/](02-perarm-features-bandit/)
  * [02a-build-perarm-bandit-locally.ipynb](02-perarm-features-bandit/02a-build-perarm-bandit-locally.ipynb)
  * [02b-build-training-image.ipynb](02-perarm-features-bandit/02b-build-training-image.ipynb)
  * [02c-accelerated-bandits.ipynb](02-perarm-features-bandit/02c-accelerated-bandits.ipynb)
  * [02d-scale-bandit-training-vertex.ipynb](02-perarm-features-bandit/02d-scale-bandit-training-vertex.ipynb)
* [03-ranking/](03-ranking/)
  * [baseline-ranking-agents.ipynb](03-ranking/baseline-ranking-agents.ipynb)
  * `TODO`
  * `TODO`
  * `TODO`
* [04-pipelines/](04-pipelines/)
  * [step_by_step_sdk_tf_agents_bandits_movie_recommendation](04-pipelines/step_by_step_sdk_tf_agents_bandits_movie_recommendation)
  * [mlops_pipeline_tf_agents_bandits_movie_recommendation](04-pipelines/mlops_pipeline_tf_agents_bandits_movie_recommendation)

