import os
from pathlib import Path

import joblib
import matplotlib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_PATH = "data/processed/heart_disease_cleaned.csv"
TARGET_COLUMN = "target"
MODEL_DIR = "models"
BEST_MODEL_PATH = "models/best_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
BEST_PIPELINE_PATH = "models/best_pipeline.pkl"
REPORT_PATH = "outputs/reports/model_selection_report.txt"
METRICS_CHART_PATH = "outputs/reports/model_metrics_comparison.png"

MLFLOW_DB_PATH = "outputs/mlflow/training_mlflow.db"
MLFLOW_ARTIFACT_DIR = "outputs/mlflow/training_artifacts"
MLFLOW_EXPERIMENT_NAME = "HeartDiseaseClassification"

RANDOM_STATE = 42


def configure_mlflow():
    db_file = Path(MLFLOW_DB_PATH).resolve()
    artifact_root = Path(MLFLOW_ARTIFACT_DIR).resolve()

    db_file.parent.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{db_file.as_posix()}"
    artifact_location = artifact_root.as_uri()

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

    if experiment is None:
        client.create_experiment(
            name=MLFLOW_EXPERIMENT_NAME,
            artifact_location=artifact_location,
        )

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print(f"MLflow tracking database: {db_file}")
    print(f"MLflow artifacts folder: {artifact_root}")


def load_dataset():
    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in {DATA_PATH}")

    x = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    return x, y


def save_report(results, best_model_name, best_params, best_auc):
    report_file = Path(REPORT_PATH)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "Model Selection and Tuning Report",
        "=================================",
        "",
        "Dataset:",
        f"- Input data: {DATA_PATH}",
        f"- Target column: {TARGET_COLUMN}",
        "",
        "Models trained:",
        "- Logistic Regression",
        "- Random Forest",
        "",
        "Tuning process:",
        "- GridSearchCV was used for hyperparameter tuning.",
        "- Stratified 5-fold cross-validation was used.",
        "- ROC-AUC was used as the primary model selection metric.",
        "",
        "Evaluation metrics:",
        "- Accuracy",
        "- Precision",
        "- Recall",
        "- ROC-AUC",
        "",
        "Results:",
    ]

    for result in results:
        lines.extend(
            [
                "",
                f"Model: {result['model_name']}",
                f"Best parameters: {result['best_params']}",
                f"CV accuracy: {result['cv_accuracy']:.4f}",
                f"CV precision: {result['cv_precision']:.4f}",
                f"CV recall: {result['cv_recall']:.4f}",
                f"CV ROC-AUC: {result['cv_roc_auc']:.4f}",
                f"Test accuracy: {result['accuracy']:.4f}",
                f"Test precision: {result['precision']:.4f}",
                f"Test recall: {result['recall']:.4f}",
                f"Test ROC-AUC: {result['roc_auc']:.4f}",
                f"Confusion matrix: {result['confusion_matrix'].tolist()}",
            ]
        )

    lines.extend(
        [
            "",
            f"Selected best model: {best_model_name}",
            f"Selected best parameters: {best_params}",
            f"Best test ROC-AUC: {best_auc:.4f}",
        ]
    )

    report_file.write_text("\n".join(lines), encoding="utf-8")
    return report_file


def save_metrics_comparison_chart(results):
    chart_file = Path(METRICS_CHART_PATH)
    chart_file.parent.mkdir(parents=True, exist_ok=True)

    chart_data = pd.DataFrame(
        [
            {
                "model_name": result["model_name"],
                "Accuracy": result["accuracy"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "ROC-AUC": result["roc_auc"],
            }
            for result in results
        ]
    ).set_index("model_name")
    
    ax = chart_data.plot(kind="bar", figsize=(10, 6), width=0.75)
    ax.set_title("Test Metrics Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Metric", loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(chart_file, dpi=150)
    plt.close()

    return chart_file

def save_reusable_pipeline(best_model):
    preprocessor_file = Path(PREPROCESSOR_PATH)

    if not preprocessor_file.exists():
        raise FileNotFoundError(
            f"Preprocessor file not found: {preprocessor_file}. "
            "Run preprocess_data.py before model_train.py."
        )

    preprocessor = joblib.load(preprocessor_file)
    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", best_model),
        ]
    )

    pipeline_file = Path(BEST_PIPELINE_PATH)
    pipeline_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(full_pipeline, pipeline_file)

    return pipeline_file, full_pipeline

def get_models_and_params():
    models = {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["liblinear"],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "n_estimators": [100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
            },
        ),
    }

    return models


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

def main():
    configure_mlflow()

    x, y = load_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }

    best_model = None
    best_model_name = None
    best_params = None
    best_auc = 0
    all_results = []

    for model_name, (model, param_grid) in get_models_and_params().items():
        with mlflow.start_run(run_name=model_name):
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
            )
            grid_search.fit(x_train, y_train)

            tuned_model = grid_search.best_estimator_

            cv_scores = cross_validate(
                tuned_model,
                x_train,
                y_train,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
            )

            test_metrics = evaluate_model(tuned_model, x_test, y_test)

            result = {
                "model_name": model_name,
                "best_params": grid_search.best_params_,
                "cv_accuracy": cv_scores["test_accuracy"].mean(),
                "cv_precision": cv_scores["test_precision"].mean(),
                "cv_recall": cv_scores["test_recall"].mean(),
                "cv_roc_auc": cv_scores["test_roc_auc"].mean(),
                **test_metrics,
            }
            all_results.append(result)

            mlflow.log_param("model_name", model_name)
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("cv_accuracy", result["cv_accuracy"])
            mlflow.log_metric("cv_precision", result["cv_precision"])
            mlflow.log_metric("cv_recall", result["cv_recall"])
            mlflow.log_metric("cv_roc_auc", result["cv_roc_auc"])
            mlflow.log_metric("test_accuracy", result["accuracy"])
            mlflow.log_metric("test_precision", result["precision"])
            mlflow.log_metric("test_recall", result["recall"])
            mlflow.log_metric("test_roc_auc", result["roc_auc"])

            mlflow.sklearn.log_model(
                sk_model=tuned_model,
                name="heart_model",
            )

            print(f"\n{model_name}")
            print(f"Best Params : {grid_search.best_params_}")
            print(f"CV ROC-AUC  : {result['cv_roc_auc']:.4f}")
            print(f"Accuracy    : {result['accuracy']:.4f}")
            print(f"Precision   : {result['precision']:.4f}")
            print(f"Recall      : {result['recall']:.4f}")
            print(f"ROC-AUC     : {result['roc_auc']:.4f}")
            print(f"Run ID      : {mlflow.active_run().info.run_id}")

            if result["roc_auc"] > best_auc:
                best_auc = result["roc_auc"]
                best_model = tuned_model
                best_model_name = model_name
                best_params = grid_search.best_params_

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, BEST_MODEL_PATH)
    pipeline_path, full_pipeline = save_reusable_pipeline(best_model)

    report_path = save_report(
        results=all_results,
        best_model_name=best_model_name,
        best_params=best_params,
        best_auc=best_auc,
    )
    metrics_chart_path = save_metrics_comparison_chart(all_results)

    with mlflow.start_run(run_name="BestModelSummary"):
        mlflow.log_param("selected_model", best_model_name)
        mlflow.log_param("selected_model_params", best_params)
        mlflow.log_metric("best_test_roc_auc", best_auc)
        mlflow.sklearn.log_model(best_model, name="best_heart_model")
        mlflow.sklearn.log_model(full_pipeline, name="best_heart_pipeline")
        mlflow.log_artifact(BEST_MODEL_PATH, artifact_path="saved_model")
        mlflow.log_artifact(str(pipeline_path), artifact_path="saved_model")
        mlflow.log_artifact(str(report_path), artifact_path="reports")
        mlflow.log_artifact(str(metrics_chart_path), artifact_path="reports")

        print("\nBest model saved successfully")
        print(f"Best model      : {best_model_name}")
        print(f"Best ROC-AUC    : {best_auc:.4f}")
        print(f"Model path      : {BEST_MODEL_PATH}")
        print(f"Pipeline path   : {pipeline_path}")
        print(f"Report path     : {report_path}")
        print(f"Metrics chart   : {metrics_chart_path}")
        print(f"MLflow run ID   : {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()