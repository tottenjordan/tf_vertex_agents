# Baseline PerArm Bandit for Movielens

Here we will start with a few, simple contextual bandit implementations, where each will make use of *per-arm* features (i.e., features describing the arms or items to recomemnd)

**Notebooks**
* [01a-build_perarm_stationary_env.ipynb](01a-build_perarm_stationary_env.ipynb) - demonstrate how to use a pre-built environment to train a policy
* [01b-build_perarm_mf_env.ipynb](01b-build_perarm_mf_env.ipynb) - demonstrate how to build a custom environment class to train a policy
* [01c-build-training-image.ipynb](01c-build-training-image.ipynb) - build a training image for Vertex AI Training
* [01d-train_perarm_tf_agents_vertex.ipynb](01d-train_perarm_tf_agents_vertex.ipynb) - Submit hptuning job, using best hpt params, launch full-scale training job (both hpt and full-training submitted to Vertex AI training)

### Managed Tensorboard

#### Hyperparameter tuning jobs

![alt text](https://github.com/tottenjordan/tf_vertex_agents/blob/main/imgs/01_hpt_tboard.png)


