# RL MLOps for RecSys

> develop contextual bandits with TF-Agents and orchstrate their MLOps with KFP; deploy and scale with Vertex AI

Here we adapt [examples from the Vertex AI community samples](https://github.com/GoogleCloudPlatform/vertex-ai-samples/tree/main/community-content/tf_agents_bandits_movie_recommendation_with_kfp_and_vertex_sdk) that demonstrate how to use TF-Agents, Kubeflow Pipelines (KFP) and Vertex AI to build RL RecSys applications

---

#### This section of the repo includes two main workstreams:

1. **Step-by-step:** showcase how to use custom training, custom hyperparameter tuning, custom prediction and endpoint deployment of Vertex AI to build an RL RecSys

2. **End-to-end MLOps pipeline:** showcase how to build an RL-specific MLOps pipeline using KFP and Vertex Pipelines, as well as additional Vertex AI and GCP services such as BigQuery, Cloud Functions, Cloud Scheduler, Pub/Sub. 

*Both workstreams contains a notebook that carries out the full workflow and user instructions, and a `src/` directory for Python modules and unit tests*

## [1] Step-by-Step RL for RecSys

> **TODO**

## [2] End-to-end MLOps RL pipeline for RecSys

> **TODO**

###  Initial training pipeline with environment simulation

![alt text](https://github.com/tottenjordan/tf_vertex_agents/blob/main/imgs/mab_mlops_pipe.png)

#### Pipeline highlights:
* RL-specific implementation for training and prediction
* Simulation for initial training data, prediction requests and re-training
* Closing of the feedback loop from prediction results back to training
* Customizable and reproducible KFP components

#### pipeline explained

* `Generator`: generates [MovieLens](https://www.kaggle.com/prajitdatta/movielens-100k-dataset) simulation data
* `Ingester`: ingests data
* `Trainer`: trains the RL policy
* `Deployer`: deploys trained policy to Vertex AI endpoint

## Continuous training pipeline

![alt text](https://github.com/tottenjordan/tf_vertex_agents/blob/main/imgs/retraining_pipeline_overview.png)

### pipeline highlights
* The re-training pipeline (executed recurrently) includes the `Ingester`, `Trainer`, and Deployment steps
* it does not need initial training data from the `Generator`

## To model production traffic we create these additional modules:
* `Simulator` for initial training data, prediction requests and re-training
* `Logger` to asynchronously log prediction inputs and results. 
* Pipeline `Trigger` to trigger recurrent re-training


When the `Simulator` sends prediction requests to the endpoint, the `Logger` is triggered by the hook in the prediction code to log prediction results to BigQuery as new training data. As this pipeline has a recurrent schedule, it utlizes the new training data in training a new policy, therefore closing the feedback loop. Theoretically speaking, if you set the pipeline scheduler to be infinitely frequent, then you would be approaching real-time, continuous training.