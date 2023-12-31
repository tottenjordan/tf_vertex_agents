# End-to-End MLOps Pipeline Demo

This demo showcase how to build a RL-specific MLOps pipeline using
[Kubeflow Pipelines (KFP)](https://www.kubeflow.org/docs/components/pipelines/overview/pipelines-overview/)
and [Vertex Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines), as
well as additional [Vertex AI](https://cloud.google.com/vertex-ai) and GCP
services such as [BigQuery](https://cloud.google.com/bigquery),
[Cloud Functions](https://cloud.google.com/functions),
[Cloud Scheduler](https://cloud.google.com/scheduler),
[Pub/Sub](https://cloud.google.com/pubsub). We implement the RL training and
prediction logic using the [TF-Agents](https://www.tensorflow.org/agents)
library.

This notebook illustrates how to orchestrate all the steps covered in the step-by-step demo, and

## (1) Creating and testing RL-specific pipeline components

* Create the `Generator` component 
  * Generates initial set of training data using a [MovieLens](https://www.kaggle.com/prajitdatta/movielens-100k-dataset) simulation [environment](https://github.com/tensorflow/agents/blob/master/tf_agents/bandits/environments/movielens_py_environment.py) and a random data-collecting policy
  * Store generated training *trajectories* in BigQuery
  * **steps:** raw dataset → environment → random policy generating (random) sets of training data samples 

* Create the `Ingester` component
  * Ingests data from BigQuery, packages them as `tf.train.Example` objects, and writes them to TFRecords
  * **steps:** BigQuery → `tf.train.example` → TFRecord

* Create the `Trainer` component
  * Trains RL policy with sumlated training data
  * Off-policy training, where policy is trained on static set of pre-collected data; includes observation, action, and reward
  * Implements the [LinUCB agent](https://www.tensorflow.org/agents/api_docs/python/tf_agents/bandits/agents/lin_ucb_agent/LinearUCBAgent) and saves the trained policy as a `SavedModel`

* Create the `Deployer` component
  * Deploys trained policy to a Vertex AI endpoint
  * Uses Custom Prediction container for Vertex Ai Prediciton
  * **steps:** upload trained policy → create Vertex AI endpoint → deploy trained policy to Verted AI Endpoint

## (2) Compiling a train-deploy pipeline that allows for continuous training

In practice, there are times we don't have sufficient training data that represents our real-world problem (e.g., initial Agent deployments, new use cases, changing label definitions, new training data filters, etc.). A common way to address this is to begin training an agent with data sampled from a simulation `environment`. Then, over time, as we log interactions and their rewards, we may choose to generate training data from these without implementing an `environmnet`.

The pipeline runs produced in this example demonstrate how we may account for both of these scenarios. (1) We initially implement a train-deploy pipeline that begins with a `generator` component responsible for creating training data. (2) We then show how to deploy the same pipeline without the `generator` component, where its assumed sufficient training data is already available. 

> *Note: this is just one set of possibilities for the sake of demonstrating MLOps concepts. Look for further discussion and details re: *environments* in `learning/`*

### Example run for the initial training pipeline with environment simulation

![alt text](https://github.com/tottenjordan/tf_vertex_agents/blob/main/imgs/mab_mlops_pipe.png)

For custom training, we implement off-policy training, using a static set of
pre-collected data records. "Off-policy" refers to the situation where for a
data record, given its observation, the current policy in training might not
choose the same action as the one in said data record.

The pipeline (startup) includes the following KFP custom components (executed
once):

-   Generator to generate MovieLens simulation data
-   Ingester to ingest data
-   Trainer to train the RL policy
-   Deployer to deploy the trained policy to a Vertex AI endpoint

### Example run for the continuous training pipeline:

![alt text](https://github.com/tottenjordan/tf_vertex_agents/blob/main/imgs/retraining_pipeline_overview.png)

The continuous / re-training pipeline (executed recurrently) includes the Ingester, Trainer
and Deployer, as it does not need initial training data from the Generator.

After pipeline construction, we also create:

-   `Simulator` (which utilizes Cloud Functions, Cloud Scheduler and Pub/Sub) to
    send simulated MovieLens prediction requests
-   `Logger` to asynchronously log prediction inputs and results (which utilizes
    Cloud Functions, Pub/Sub and a hook in the prediction code)
-   `Trigger` to trigger recurrent re-training.


## Conceptual: RL MLOps Pipeline Design

![alt text](https://github.com/tottenjordan/tf_vertex_agents/blob/main/imgs/mlops_pipeline_design.png)

The demo contains a notebook that carries out the full workflow and user
instructions, and a `src/` directory for Python modules and unit tests.

Read more about problem framing, simulations, and adopting this demo in
production and to other use cases in the notebook.

# Misc. topics

## Pipeline Caching?

When Vertex AI Pipelines runs a pipeline, it checks to see whether or not an execution exists in Vertex ML Metadata with the interface (cache key) of each pipeline step.

The step's interface is defined as the combination of the following:
* The pipeline step's inputs. These inputs include the input parameters' value (if any) and the input artifact id (if any).
* The pipeline step's output definition. This output definition includes output parameter definition (name, if any) and output artifact definition (name, if any).
* The component's specification. This specification includes the image, commands, arguments and environment variables being used, as well as the order of the commands and arguments.

**Only pipelines with the same `pipeline-name` will share the cache** 
* This means we can easily manipulate the caching by including a prefix or tag in the pipeline name
* This repo uses the string identifiers `PREFIX` and `PIPE_VERSION` to organize a set of runs / experiments. These are used to name most of the produced artifacts. this helps with tracking over many pipeline runs and experiments

If there is a matching execution in Vertex ML Metadata, the outputs of that execution are used and the step is skipped. This helps to reduce costs by skipping computations that were completed in a previous pipeline run.
