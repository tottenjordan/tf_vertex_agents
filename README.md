# Vertex Agents & Bandits

> Contextual Bandits for RecSys with TF Agents and Vertex AI 

## How to use this repo

This repo is organized across various notebooks:
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
  
### Objectives
* [00-env-setup.ipynb](00-env-setup.ipynb) - create variables names and env config to run across notebooks. (easier tracking across notebooks)
* [01-movielens-data-prep.ipynb](01-movielens-data-prep.ipynb) - prepare standard movielens dataset, and a listwise version for ranking
* [01-baseline-perarm-bandit/](01-baseline-perarm-bandit/) - simulation environment for multi-armed bandit with arm features
* [02-perarm-features-bandit/](02-perarm-features-bandit/) - train contextual bandits with per-arm features
* [03-ranking/](03-ranking/) - training contextual bandits for ranking problems
* [04-pipelines/](04-pipelines/) - mlops pipeline for multi-armed bandit

## What are per-arm features?
* Many agents implemented in `tf-agents` can run on environments that have features for its actions 
* these environments are feferred to as “per-arm environments”
* ability to process additional context features --> potentially better solution for learning nonlinear patterns
* Reward is modeled not per-arm, but globally

> TODO: include visual illustrating difference between (1) multi-armed bandita and (2) contextual bandit with per-arm features

## Contextual Bandit Ranking

> TODO


## Profiling Agents

> TODO

![alt text](https://github.com/tottenjordan/tf_vertex_agents/blob/main/imgs/getting_profiler.png)

### Input pipeline bottleneck analysis

> TODO

![alt text](https://github.com/tottenjordan/tf_vertex_agents/blob/main/imgs/tb_input_bottleneck_analysis.png)

