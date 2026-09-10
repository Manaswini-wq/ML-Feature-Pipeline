"""
Ingest raw customer data into BigQuery.
Supports CSV upload or synthetic data generation for demo.
"""

import pandas as pd
import numpy as np
from google.cloud import bigquery
from src.utils import load_config, get_env


def generate_synthetic_data(n=10000):
    """Generate synthetic telecom customer data for churn prediction."""
    np.random.seed(42)
    data = {
        "customer_id": [f"CUST_{i:06d}" for i in range(n)],
        "tenure_months": np.random.randint(1, 72, n),
        "monthly_charges": np.round(np.random.uniform(20, 120, n), 2),
        "total_charges": np.zeros(n),
        "contract_type": np.random.choice(
            ["month-to-month", "one_year", "two_year"], n, p=[0.5, 0.3, 0.2]
        ),
        "payment_method": np.random.choice(
            ["credit_card", "bank_transfer", "electronic_check", "mailed_check"], n
        ),
        "num_support_tickets": np.random.poisson(2, n),
        "num_referrals": np.random.poisson(1, n),
        "internet_service": np.random.choice(
            ["fiber_optic", "dsl", "none"], n, p=[0.5, 0.35, 0.15]
        ),
        "online_security": np.random.choice([0, 1], n),
        "tech_support": np.random.choice([0, 1], n),
        "streaming_tv": np.random.choice([0, 1], n),
        "streaming_movies": np.random.choice([0, 1], n),
    }
    df = pd.DataFrame(data)
    df["total_charges"] = np.round(
        df["tenure_months"] * df["monthly_charges"] * np.random.uniform(0.85, 1.0, n), 2
    )

    # Churn logic: higher churn for month-to-month, high charges, low tenure
    churn_prob = (
        0.3 * (df["contract_type"] == "month-to-month").astype(float)
        + 0.2 * (df["monthly_charges"] > 70).astype(float)
        + 0.2 * (df["tenure_months"] < 12).astype(float)
        + 0.15 * (df["num_support_tickets"] > 3).astype(float)
        + 0.1 * np.random.uniform(0, 1, n)
    )
    df["churned"] = (churn_prob > 0.5).astype(int)

    return df


def upload_to_bigquery(df):
    config = load_config()
    project = get_env("GCP_PROJECT_ID")
    table_id = f"{project}.{config['bigquery']['dataset']}.{config['bigquery']['raw_table']}"

    client = bigquery.Client(project=project)

    dataset_ref = bigquery.DatasetReference(project, config["bigquery"]["dataset"])
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = config["gcp"]["region"]
    client.create_dataset(dataset, exists_ok=True)

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    print(f"Uploaded {len(df)} rows to {table_id}")


if __name__ == "__main__":
    df = generate_synthetic_data()
    print(f"Generated {len(df)} rows, churn rate: {df['churned'].mean():.2%}")
    upload_to_bigquery(df)
