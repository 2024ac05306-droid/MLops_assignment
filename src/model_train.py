import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    roc_auc_score,
    confusion_matrix
)   


# ---------------------------------------------------
# MLflow Configuration
# ---------------------------------------------------

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("HeartDiseaseClassification")

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

df = pd.read_csv(
    "data/processed/heart_disease_cleaned.csv"
)

X = df.drop("target", axis=1)
y = df["target"]

# ---------------------------------------------------
# Train-Test Split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------
# Models
# ---------------------------------------------------

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

best_model = None
best_auc = 0

# ---------------------------------------------------
# Training Loop
# ---------------------------------------------------

for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name):

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(y_test, y_pred)

        recall = recall_score(y_test, y_pred)

        roc_auc = roc_auc_score(y_test, y_prob)

        # ----------------------------------------
        # MLflow Logging
        # ----------------------------------------

        mlflow.log_param(
            "model_name",
            model_name
        )

        if model_name == "RandomForest":

            mlflow.log_param(
                "n_estimators",
                100
            )

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )

        mlflow.sklearn.log_model(
            sk_model = model,
            name="heart_model"
        )

        print(f"\n{model_name}")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")

        # Save Best Model
        if roc_auc > best_auc:

            best_auc = roc_auc
            best_model = model

# ---------------------------------------------------
# Save Best Model
# ---------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print("\nBest model saved successfully")

