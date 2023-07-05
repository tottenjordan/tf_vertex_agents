# Vertex Agents & Bandits
training and serving TF Agents - Bandits with Vertex AI 

### Using RL for recommendations
* User vectors are the environment observations
* Items to recommend are the agent actions applied on the environment
* Approximate user ratings are the environment rewards generated as feedback to the observations and actions

For custom training, we implement **off-policy training**, using a static set of pre-collected data records. "Off-policy" refers to the situation where for a data record, given its observation, the current policy in training might not choose the same action as the one in said data record.

## End-to-end MLOps pipeline for RL-specific implementations
![](img/e2e_rl_pipeline.png)

### pipeline highlights
* `Generator`: generates [MovieLens](https://www.kaggle.com/prajitdatta/movielens-100k-dataset) simulation data
* `Ingester`: ingests data
* `Trainer`: trains the RL policy
* Deploying trained policy to Vertex AI endpoint

## (re)Training pipeline for RL-specific implementations
![](img/retraining_pipeline_overview.png)

### pipeline highlights
* The re-training pipeline (executed recurrently) includes the `Ingester`, `Trainer`, and Deployment steps
* it does not need initial training data from the `Generator`

## To model production traffic we create these additional modules:
* `Simulator` for initial training data, prediction requests and re-training
* `Logger` to asynchronously log prediction inputs and results. 
* Pipeline `Trigger` to trigger recurrent re-training


When the `Simulator` sends prediction requests to the endpoint, the `Logger` is triggered by the hook in the prediction code to log prediction results to BigQuery as new training data. As this pipeline has a recurrent schedule, it utlizes the new training data in training a new policy, therefore closing the feedback loop. Theoretically speaking, if you set the pipeline scheduler to be infinitely frequent, then you would be approaching real-time, continuous training.
