
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.dsl import (
    component, 
    Metrics
)
from . import pipeline_config

@component(
    base_image=pipeline_config.DATA_PIPELINE_IMAGE,
    install_kfp_package=False
)
def write_tf_records(
    project_id: str,
    location: str,
    pipeline_version: str,
    bq_table_ref: str,
    tf_record_file: str,
    global_dim: int,
    per_arm_dim: int,
) -> NamedTuple('Outputs', [
    ('tf_record_file', str),
    ('global_dim', int),
    ('per_arm_dim', int),
    ('bq_table_ref', str),
]):
    
    from google.cloud import bigquery
    
    # this repo
    from src.data import data_utils as data_utils
    from src.data_preprocessor import preprocess_utils
    
    bqclient = bigquery.Client(project=project_id)
    
    # get bq table iterator
    print(f"getting bq table iterator...")
    
    bq_table = bqclient.get_table(bq_table_ref)
    print(f"Got table: `{bq_table.project}.{bq_table.dataset_id}.{bq_table.table_id}`")
    print("Table has {} rows".format(bq_table.num_rows))

    table_row_iter = bqclient.list_rows(bq_table)
    
    print(f"writting bq to tf records...")
    preprocess_utils.write_tfrecords(tf_record_file, table_row_iter)
    
    print(f"tf record complete: {tf_record_file}")
    
    return (
        f'{tf_record_file}',
        global_dim,
        per_arm_dim,
        bq_table_ref,
    )
