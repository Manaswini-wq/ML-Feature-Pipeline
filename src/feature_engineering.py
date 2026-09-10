"""
PySpark feature engineering pipeline.
Reads raw data, computes derived features, writes to feature store.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from src.utils import load_config, get_env


def create_spark_session():
    return (
        SparkSession.builder
        .appName("ChurnFeatureEngineering")
        .config("spark.jars.packages", "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.36.1")
        .getOrCreate()
    )


def read_raw_data(spark, project, dataset, table):
    return (
        spark.read
        .format("bigquery")
        .option("project", project)
        .option("table", f"{project}.{dataset}.{table}")
        .load()
    )


def engineer_features(df):
    """Apply feature transformations."""

    # 1. Avg monthly charge ratio
    df = df.withColumn(
        "avg_charge_per_month",
        F.when(F.col("tenure_months") > 0,
               F.col("total_charges") / F.col("tenure_months"))
        .otherwise(F.col("monthly_charges"))
    )

    # 2. Charge deviation from average
    df = df.withColumn(
        "charge_deviation",
        F.col("monthly_charges") - F.col("avg_charge_per_month")
    )

    # 3. Support ticket rate per month
    df = df.withColumn(
        "ticket_rate",
        F.when(F.col("tenure_months") > 0,
               F.col("num_support_tickets") / F.col("tenure_months"))
        .otherwise(F.lit(0.0))
    )

    # 4. Services count
    df = df.withColumn(
        "total_services",
        F.col("online_security") + F.col("tech_support")
        + F.col("streaming_tv") + F.col("streaming_movies")
    )

    # 5. Tenure bucket
    df = df.withColumn(
        "tenure_bucket",
        F.when(F.col("tenure_months") < 12, "new")
        .when(F.col("tenure_months") < 36, "mid")
        .otherwise("loyal")
    )

    # 6. Is high value customer
    df = df.withColumn(
        "is_high_value",
        F.when(F.col("total_charges") > 3000, 1).otherwise(0)
    )

    # 7. Has premium services
    df = df.withColumn(
        "has_premium",
        F.when(
            (F.col("internet_service") == "fiber_optic") & (F.col("total_services") >= 3),
            1
        ).otherwise(0)
    )

    # 8. One-hot encode contract type
    for contract in ["month-to-month", "one_year", "two_year"]:
        col_name = f"contract_{contract.replace('-', '_')}"
        df = df.withColumn(
            col_name,
            F.when(F.col("contract_type") == contract, 1).otherwise(0)
        )

    # 9. One-hot encode internet service
    for svc in ["fiber_optic", "dsl", "none"]:
        df = df.withColumn(
            f"internet_{svc}",
            F.when(F.col("internet_service") == svc, 1).otherwise(0)
        )

    # 10. Referral-to-ticket ratio
    df = df.withColumn(
        "referral_ticket_ratio",
        F.when(F.col("num_support_tickets") > 0,
               F.col("num_referrals") / F.col("num_support_tickets"))
        .otherwise(F.col("num_referrals").cast("double"))
    )

    # Add feature version timestamp
    df = df.withColumn("feature_timestamp", F.current_timestamp())

    return df


def write_features(df, project, dataset, table):
    (
        df.write
        .format("bigquery")
        .option("table", f"{project}.{dataset}.{table}")
        .option("temporaryGcsBucket", f"{project}-dataflow-temp")
        .mode("overwrite")
        .save()
    )


def run():
    config = load_config()
    project = get_env("GCP_PROJECT_ID")
    spark = create_spark_session()

    print("Reading raw data...")
    raw_df = read_raw_data(
        spark, project,
        config["bigquery"]["dataset"],
        config["bigquery"]["raw_table"],
    )
    print(f"Raw rows: {raw_df.count()}")

    print("Engineering features...")
    feature_df = engineer_features(raw_df)

    feature_cols = [
        "customer_id", "tenure_months", "monthly_charges", "total_charges",
        "num_support_tickets", "num_referrals",
        "avg_charge_per_month", "charge_deviation", "ticket_rate",
        "total_services", "is_high_value", "has_premium",
        "contract_month_to_month", "contract_one_year", "contract_two_year",
        "internet_fiber_optic", "internet_dsl", "internet_none",
        "online_security", "tech_support", "streaming_tv", "streaming_movies",
        "referral_ticket_ratio", "tenure_bucket",
        "churned", "feature_timestamp",
    ]
    feature_df = feature_df.select(feature_cols)

    print("Writing to feature store...")
    write_features(
        feature_df, project,
        config["bigquery"]["dataset"],
        config["bigquery"]["feature_table"],
    )
    print(f"Wrote {feature_df.count()} feature rows.")
    spark.stop()


if __name__ == "__main__":
    run()
