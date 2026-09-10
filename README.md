# ML Feature Pipeline — Customer Churn Prediction

An end-to-end ML feature engineering pipeline that ingests raw customer data, transforms it with PySpark, stores features in a BigQuery-based feature store, trains a churn prediction model, and serves real-time predictions via a FastAPI endpoint.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Raw Data   │────→│  PySpark Feature │────→│  BigQuery        │
│  (CSV/API)  │     │  Engineering     │     │  Feature Store   │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                      │
                                              ┌───────▼────────┐
                                              │  Model Training │
                                              │  (scikit-learn) │
                                              └───────┬────────┘
                                                      │
                                              ┌───────▼────────┐
                                              │  FastAPI        │
                                              │  Prediction API │
                                              └────────────────┘
```

## Tech Stack

| Component            | Technology                        |
|----------------------|-----------------------------------|
| Feature Engineering  | PySpark                           |
| Feature Store        | Google BigQuery                   |
| Data Ingestion       | Python, Pandas                    |
| Model Training       | scikit-learn (GradientBoosting)   |
| Model Serving        | FastAPI, Uvicorn                  |
| Infrastructure       | Terraform, Docker                 |
| Cloud Platform       | Google Cloud (GCP)                |

## Features Engineered

| Feature                  | Description                                          |
|--------------------------|------------------------------------------------------|
| avg_charge_per_month     | Total charges / tenure                               |
| charge_deviation         | Monthly charges vs average deviation                 |
| ticket_rate              | Support tickets per month of tenure                  |
| total_services           | Count of active services (security, streaming, etc.) |
| is_high_value            | Total charges > $3000                                |
| has_premium              | Fiber optic + 3 or more services                     |
| referral_ticket_ratio    | Referrals / support tickets                          |
| tenure_bucket            | new (<12mo), mid (12-36mo), loyal (36mo+)            |
| contract_*               | One-hot encoded contract type                        |
| internet_*               | One-hot encoded internet service                     |

## Project Structure

```
ml-feature-pipeline/
├── config/
│   └── config.yaml              # All project configuration
├── src/
│   ├── utils.py                 # Config loader, env helpers
│   ├── ingest.py                # Raw data generation + BigQuery upload
│   ├── feature_engineering.py   # PySpark feature transforms
│   ├── feature_store.py         # BigQuery feature store read/write
│   ├── train.py                 # Model training + evaluation
│   └── predict_api.py           # FastAPI serving endpoint
├── models/
│   └── churn_model.joblib       # Trained model artifact
├── tests/
│   └── test_features.py         # PySpark feature tests
├── terraform/
│   └── main.tf                  # GCP infrastructure
├── Dockerfile
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.11+
- Java 17 (for PySpark)
- GCP account with BigQuery and Storage enabled
- Reddit API credentials (for streaming project integration)

## Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/ml-feature-pipeline.git
cd ml-feature-pipeline

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your GCP project ID and credentials path

# Provision GCP resources (optional — uses Terraform)
cd terraform
terraform init
terraform apply -var="project_id=your-project-id"
cd ..
```

## Usage

### Step 1: Ingest raw data
```bash
python -m src.ingest
```
Generates 10,000 synthetic customer records and uploads to BigQuery.

### Step 2: Run feature engineering
```bash
python -m src.feature_engineering
```
Reads raw data, computes 10+ derived features using PySpark, writes to feature store.

### Step 3: Train model
```bash
python -m src.train
```
Loads features from BigQuery, trains a GradientBoosting classifier, prints evaluation metrics, saves model to `models/churn_model.joblib`.

### Step 4: Serve predictions
```bash
python -m src.predict_api
```
Starts FastAPI server at `http://localhost:8000`.

### Step 5: Test predictions
```bash
# Predict by customer ID (fetches features from feature store)
curl -X POST http://localhost:8000/predict/customer \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "CUST_000042"}'

# Predict by raw features
curl -X POST http://localhost:8000/predict/features \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 6,
    "monthly_charges": 95.0,
    "total_charges": 570.0,
    "num_support_tickets": 5,
    "num_referrals": 0,
    "avg_charge_per_month": 95.0,
    "charge_deviation": 0.0,
    "ticket_rate": 0.83,
    "total_services": 1,
    "is_high_value": 0,
    "has_premium": 0,
    "contract_month_to_month": 1,
    "contract_one_year": 0,
    "contract_two_year": 0,
    "internet_fiber_optic": 1,
    "internet_dsl": 0,
    "internet_none": 0,
    "online_security": 0,
    "tech_support": 0,
    "streaming_tv": 1,
    "streaming_movies": 0,
    "referral_ticket_ratio": 0.0
  }'
```

### API Response
```json
{
  "customer_id": "CUST_000042",
  "churn_probability": 0.7823,
  "will_churn": true,
  "risk_level": "high"
}
```

### API Docs
Interactive Swagger docs available at `http://localhost:8000/docs`

## Run Tests

```bash
pytest tests/ -v
```

## Docker

```bash
# Build
docker build -t ml-feature-pipeline .

# Run API
docker run -p 8000:8000 \
  -e GCP_PROJECT_ID=your-project-id \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/sa.json \
  -v /path/to/sa.json:/app/sa.json \
  ml-feature-pipeline
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| BigQuery as feature store | Serverless, scales to TB, native GCP integration |
| PySpark for features | Handles large-scale transforms, portable to Dataproc |
| GradientBoosting | Strong baseline for tabular churn data |
| FastAPI for serving | Async, auto-docs, production-grade performance |
| Synthetic data | Reproducible demo without PII concerns |

## Future Improvements

- [ ] Add Airflow DAG for scheduled feature refresh
- [ ] Implement feature versioning with timestamps
- [ ] Add model registry (Vertex AI Model Registry)
- [ ] A/B testing framework for model comparison
- [ ] Monitoring dashboard for feature drift detection
- [ ] CI/CD pipeline with GitHub Actions

## License

MIT
