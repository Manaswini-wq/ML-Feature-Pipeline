"""Train a churn prediction model from feature store data."""

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from src.feature_store import get_training_features
from src.utils import load_config


def train():
    config = load_config()
    target = config["model"]["target_column"]
    model_path = config["model"]["path"]

    print("Loading features from feature store...")
    df = get_training_features()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["model"]["test_size"], random_state=42, stratify=y
    )

    print("Training model...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    # Feature importance
    importance = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)
    print("\n--- Top 10 Features ---")
    print(importance.head(10))

    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    train()
