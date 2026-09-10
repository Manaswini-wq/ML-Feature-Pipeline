"""Read features from BigQuery feature store for training or serving."""

import pandas as pd
from google.cloud import bigquery
from src.utils import load_config, get_env


def get_training_features():
    config = load_config()
    project = get_env("GCP_PROJECT_ID")
    dataset = config["bigquery"]["dataset"]
    table = config["bigquery"]["feature_table"]

    client = bigquery.Client(project=project)
    query = f"""
        SELECT * EXCEPT(customer_id, tenure_bucket, feature_timestamp)
        FROM `{project}.{dataset}.{table}`
    """
    return client.query(query).to_dataframe()


def get_customer_features(customer_id):
    config = load_config()
    project = get_env("GCP_PROJECT_ID")
    dataset = config["bigquery"]["dataset"]
    table = config["bigquery"]["feature_table"]

    client = bigquery.Client(project=project)
    query = f"""
        SELECT * EXCEPT(customer_id, churned, tenure_bucket, feature_timestamp)
        FROM `{project}.{dataset}.{table}`
        WHERE customer_id = @cid
        ORDER BY feature_timestamp DESC
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("cid", "STRING", customer_id)
        ]
    )
    df = client.query(query, job_config=job_config).to_dataframe()
    if df.empty:
        return None
    return df.iloc[0].to_dict()
