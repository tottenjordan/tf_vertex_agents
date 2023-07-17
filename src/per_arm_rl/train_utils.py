
import numpy as np
import tensorflow as tf
import logging
logging.disable(logging.WARNING)

from google.cloud import storage

def _is_chief(task_type, task_id): 
    ''' Check for primary if multiworker training
    '''
    if task_type == 'chief':
        results = 'chief'
    else:
        results = None
    return results

def get_train_strategy(distribute_arg):

    # Single Machine, single compute device
    if distribute_arg == 'single':
        if tf.config.list_physical_devices('GPU'):
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        else:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        logging.info("Single device training")  
    # Single Machine, multiple compute device
    elif distribute_arg == 'mirrored':
        strategy = tf.distribute.MirroredStrategy()
        logging.info("Mirrored Strategy distributed training")
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    # Multi Machine, multiple compute device
    elif distribute_arg == 'multiworker':
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        logging.info("Multi-worker Strategy distributed training")
        logging.info('TF_CONFIG = {}'.format(os.environ.get('TF_CONFIG', 'Not found')))
    # Single Machine, multiple TPU devices
    elif distribute_arg == 'tpu':
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)
        logging.info("All devices: ", tf.config.list_logical_devices('TPU'))

    return strategy

def prepare_worker_pool_specs(
    image_uri,
    args,
    replica_count=1,
    machine_type="n1-standard-16",
    accelerator_count=1,
    accelerator_type="ACCELERATOR_TYPE_UNSPECIFIED",
    reduction_server_count=0,
    reduction_server_machine_type="n1-highcpu-16",
    reduction_server_image_uri="us-docker.pkg.dev/vertex-ai-restricted/training/reductionserver:latest",
):

    if accelerator_count > 0:
        machine_spec = {
            "machine_type": machine_type,
            "accelerator_type": accelerator_type,
            "accelerator_count": accelerator_count,
        }
    else:
        machine_spec = {"machine_type": machine_type}

    container_spec = {
        "image_uri": image_uri,
        "args": args,
    }

    chief_spec = {
        "replica_count": 1,
        "machine_spec": machine_spec,
        "container_spec": container_spec,
    }

    worker_pool_specs = [chief_spec]
    if replica_count > 1:
        workers_spec = {
            "replica_count": replica_count - 1,
            "machine_spec": machine_spec,
            "container_spec": container_spec,
        }
        worker_pool_specs.append(workers_spec)
    if reduction_server_count > 1:
        workers_spec = {
            "replica_count": reduction_server_count,
            "machine_spec": {
                "machine_type": reduction_server_machine_type,
            },
            "container_spec": {"image_uri": reduction_server_image_uri},
        }
        worker_pool_specs.append(workers_spec)

    return worker_pool_specs
