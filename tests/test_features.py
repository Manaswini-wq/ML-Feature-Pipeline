from pyspark.sql import SparkSession
from src.feature_engineering import engineer_features
import pytest


@pytest.fixture(scope="session")
def spark():
    s = SparkSession.builder.master("local[*]").appName("test").getOrCreate()
    yield s
    s.stop()


def test_feature_columns(spark):
    data = [{
        "customer_id": "C001", "tenure_months": 24, "monthly_charges": 80.0,
        "total_charges": 1900.0, "contract_type": "month-to-month",
        "payment_method": "credit_card", "num_support_tickets": 3,
        "num_referrals": 1, "internet_service": "fiber_optic",
        "online_security": 1, "tech_support": 0, "streaming_tv": 1,
        "streaming_movies": 1, "churned": 0,
    }]
    df = spark.createDataFrame(data)
    result = engineer_features(df)

    cols = result.columns
    assert "avg_charge_per_month" in cols
    assert "ticket_rate" in cols
    assert "total_services" in cols
    assert "is_high_value" in cols
    assert "contract_month_to_month" in cols
    assert "referral_ticket_ratio" in cols


def test_tenure_bucket(spark):
    data = [
        {"customer_id": "C1", "tenure_months": 6, "monthly_charges": 50.0,
         "total_charges": 300.0, "contract_type": "one_year",
         "payment_method": "bank_transfer", "num_support_tickets": 0,
         "num_referrals": 0, "internet_service": "dsl",
         "online_security": 0, "tech_support": 0, "streaming_tv": 0,
         "streaming_movies": 0, "churned": 0},
        {"customer_id": "C2", "tenure_months": 48, "monthly_charges": 90.0,
         "total_charges": 4300.0, "contract_type": "two_year",
         "payment_method": "credit_card", "num_support_tickets": 1,
         "num_referrals": 5, "internet_service": "fiber_optic",
         "online_security": 1, "tech_support": 1, "streaming_tv": 1,
         "streaming_movies": 1, "churned": 0},
    ]
    df = spark.createDataFrame(data)
    result = engineer_features(df).select("customer_id", "tenure_bucket").collect()
    buckets = {row["customer_id"]: row["tenure_bucket"] for row in result}
    assert buckets["C1"] == "new"
    assert buckets["C2"] == "loyal"


def test_services_count(spark):
    data = [{
        "customer_id": "C1", "tenure_months": 12, "monthly_charges": 60.0,
        "total_charges": 720.0, "contract_type": "one_year",
        "payment_method": "credit_card", "num_support_tickets": 1,
        "num_referrals": 2, "internet_service": "dsl",
        "online_security": 1, "tech_support": 1, "streaming_tv": 0,
        "streaming_movies": 1, "churned": 0,
    }]
    df = spark.createDataFrame(data)
    result = engineer_features(df).select("total_services").collect()
    assert result[0]["total_services"] == 3
