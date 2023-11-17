# Online simulation for training Contextual Bandits

> *Simulate* a real-world interaction environment of users and their respective preferences. To illustrate this, we take a partially labeled dataset (i.e., a dataset with feedback for a subset of `<user, item>` pairs), and create an environment that approximates feedback/rewards for all `<user, item>` pairs

## Objectives

  * `01a-train-bandit-mf-env-simulation.ipynb` - Train bandit with environment generating training data with approximated rewards
  * `01b-build-training-image.ipynb` - Build docker image for scaling training with Vertex AI
  * `01c-scale-bandit-simulation-vertex.ipynb` - Submit hptuning job, using best hpt params, launch full-scale training job (both hpt and full-training submitted to Vertex AI training)
  
## Why environment simulation?
To evaluate the performance of your RL model, you may need to run offline simulation first to determine if your RL model meets production criteria. In this case, you may have a static dataset, similar to the MovieLens dataset but potentially larger, and you can construct a custom simulation environment to use in place of the MovieLens one. In the custom environment, you may decide how to formulate observations and rewards, such as in terms of how to represent users with user vectors and what those vectors look like, perhaps via an embedding layer in a neural network. You may apply the rest of the steps and code in this demo just as you did for MovieLens, and then evaluate your model. After offline simulation, you may proceed to the next-steps of launching your model, such as A/B testing.

## Our custom environment

> TODO

the MF-based environment simulates real-world environment containing users and their respective preferences. Internally, the MovieLens simulation environment takes the user-by-movie-item rating matrix and performs a `RANK_K` matrix factorization on the rating matrix, in order to address the sparsity of the matrix. 
* After this construction step, the environment can generate user vectors of dimension `RANK_K` to represent users in the simulation environment, and is able to determine the approximate reward for any user and movie item pair. 
* In RL's language, user vectors are observations, recommended movie items are actions, and approximate ratings are rewards. 

This environment therefore defines the RL problem at hand: 

> how to recommend movies that maximize user ratings, in a simulated world of users with their respective preferences defined by the MovieLens dataset, while having zero knowledge of the internal mechanism of the environment

### Managed Tensorboard

> TODO

#### Hyperparameter tuning jobs

![alt text](https://github.com/tottenjordan/tf_vertex_agents/blob/main/imgs/01_hpt_tboard.png)


