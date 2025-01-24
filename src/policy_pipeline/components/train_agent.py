import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp import dsl
from . import pipeline_config

@dsl.component(
    base_image=pipeline_config.POLICY_PIPE_IMAGE,
    install_kfp_package=False
)
def train_agent(
    project_id: str,
    location: str,
    pipeline_version: str,
    bucket_name: str,
    example_gen_gcs_path: str,
    tfrecord_name: str,
    hparams: str,
    experiment_name: str,
    experiment_run_tag: str,
    tensorboard_resource_name: str,
    service_account: str,
    # train job
    num_epochs: int,
    log_interval: int,
    total_train_take: int,
    total_train_skip: int,
    gpi_image_name: str, # TODO
    # train compute
    replica_count: int,
    machine_type: str,
    accelerator_count: int, 
    accelerator_type: str,
    # train_ds: dsl.Input[google.VertexDataset], # google.VertexDataset
) -> NamedTuple('Outputs', [
    ('base_output_dir', str),
    ('log_dir', str),
    ('artifacts_dir', str),
]):
    # imports
    import os
    import time
    import logging
    logging.disable(logging.WARNING)
    
    # this repo
    from src.utils import train_utils
    
    from google.cloud import aiplatform, storage
    
    # GCP clients
    aiplatform.init(
        project=project_id, 
        location=location,
        # experiment=experiment_name
    )
    storage_client = storage.Client(project=project_id)
    
    # dataset
    TFRECORD_FILE = (
        f"gs://{bucket_name}/{example_gen_gcs_path}/{tfrecord_name}/{tfrecord_name}.tfrecord"
    )
    # experiment
    invoke_time       = time.strftime("%Y%m%d-%H%M%S")
    RUN_NAME          = f'{experiment_run_tag}-{invoke_time}'
    EXPERIMENT_DIR    = os.path.join(f"gs://{bucket_name}", experiment_name)
    BASE_OUTPUT_DIR   = os.path.join(EXPERIMENT_DIR, RUN_NAME)
    CHECKPT_DIR       = os.path.join(BASE_OUTPUT_DIR, "chkpoint")
    LOG_DIR           = os.path.join(BASE_OUTPUT_DIR, "logs")
    ROOT_DIR          = os.path.join(BASE_OUTPUT_DIR, "root")
    ARTIFACTS_DIR     = os.path.join(BASE_OUTPUT_DIR, "artifacts")
    # CHECKPT_DIR       = f"gs://{bucket_name}/{experiment_name}/chkpoint"
    # BASE_OUTPUT_DIR   = f"gs://{bucket_name}/{experiment_name}/{RUN_NAME}"
    # LOG_DIR           = f"{BASE_OUTPUT_DIR}/logs"
    # ARTIFACTS_DIR     = f"{BASE_OUTPUT_DIR}/artifacts"
    
    # job config 
    JOB_NAME = f'train-{experiment_name}-{experiment_run_tag}'
    logging.info(f'JOB_NAME: {JOB_NAME}')
    
    TF_GPU_THREAD_COUNT   = '4'      # '1' | '4' | '8'
    
    WORKER_ARGS = [
        f"--project={project_id}"
        , f"--location={location}"
        , f"--bucket_name={bucket_name}"
        , f"--experiment_name={experiment_name}"
        , f"--experiment_run={RUN_NAME}"
        , f"--log_dir={LOG_DIR}"
        , f"--artifacts_dir={ARTIFACTS_DIR}"
        , f"--chkpoint_dir={CHECKPT_DIR}"
        , f"--hparams={hparams}"
        ### job config
        , f"--num_epochs={num_epochs}"
        , f"--tf_record_file={TFRECORD_FILE}"
        , f"--log_interval={log_interval}"
        , f"--total_train_take={total_train_take}"
        , f"--total_train_skip={total_train_skip}"
        ### performance
        , f"--tf_gpu_thread_count={TF_GPU_THREAD_COUNT}"
        , f"--use_gpu"
        # , f"--use_tpu"
        , f"--cache_train_data"
    ]
    
    WORKER_POOL_SPECS = train_utils.prepare_worker_pool_specs(
        image_uri=f"{gpi_image_name}:latest",
        args=WORKER_ARGS,
        replica_count=replica_count,
        machine_type=machine_type,
        accelerator_count=accelerator_count,
        accelerator_type=accelerator_type,
        reduction_server_count=0,
        reduction_server_machine_type="n1-highcpu-16",
    )
    logging.info(f'WORKER_POOL_SPECS: {WORKER_POOL_SPECS}')
    
    #start the timer and training
    job = aiplatform.CustomJob(
        display_name=JOB_NAME,
        worker_pool_specs=WORKER_POOL_SPECS,
        base_output_dir=BASE_OUTPUT_DIR,
        staging_bucket=f"{BASE_OUTPUT_DIR}/staging",
    )
    logging.info(f'Submitting train job to Vertex AI...')
    job.run(
        # tensorboard=tensorboard_resource_name,
        service_account=f'{service_account}',
        restart_job_on_worker_restart=False,
        enable_web_access=True,
        sync=True,
    )
    # wait for job to complete
    # job.wait()
    
    train_job_dict = job.to_dict()
    logging.info(f'train_job_dict: {train_job_dict}')
    
    return (
        f'{BASE_OUTPUT_DIR}',
        f'{LOG_DIR}',
        f'{ARTIFACTS_DIR}',
    )
