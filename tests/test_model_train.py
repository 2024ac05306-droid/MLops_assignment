import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import model_train


def processed_training_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "numeric__age": [0.1, 0.2, 1.1, 1.2, -0.4, -0.3],
            "numeric__chol": [0.4, 0.3, 1.0, 1.1, -0.2, -0.1],
            "categorical__sex_0": [1, 1, 0, 0, 1, 0],
            "categorical__sex_1": [0, 0, 1, 1, 0, 1],
            "target": [0, 0, 1, 1, 0, 1],
        }
    )


def test_load_dataset_splits_features_and_target(tmp_path, monkeypatch):
    data_path = tmp_path / "processed.csv"
    processed_training_data().to_csv(data_path, index=False)
    monkeypatch.setattr(model_train, "DATA_PATH", str(data_path))

    x, y = model_train.load_dataset()

    assert "target" not in x.columns
    assert y.tolist() == [0, 0, 1, 1, 0, 1]


def test_load_dataset_raises_when_target_is_missing(tmp_path, monkeypatch):
    data_path = tmp_path / "processed.csv"
    processed_training_data().drop(columns=["target"]).to_csv(data_path, index=False)
    monkeypatch.setattr(model_train, "DATA_PATH", str(data_path))

    with pytest.raises(ValueError, match="Target column"):
        model_train.load_dataset()


def test_get_models_and_params_contains_expected_estimators():
    models = model_train.get_models_and_params()

    assert set(models) == {"LogisticRegression", "RandomForest"}
    for estimator, param_grid in models.values():
        assert hasattr(estimator, "fit")
        assert param_grid


def test_evaluate_model_returns_classification_metrics():
    x = processed_training_data().drop(columns=["target"])
    y = processed_training_data()["target"]
    model = LogisticRegression(max_iter=200, solver="liblinear").fit(x, y)

    metrics = model_train.evaluate_model(model, x, y)

    assert set(metrics) == {"accuracy", "precision", "recall", "roc_auc", "confusion_matrix"}
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
    assert metrics["confusion_matrix"].shape == (2, 2)


def test_save_report_writes_model_selection_summary(tmp_path, monkeypatch):
    report_path = tmp_path / "reports" / "model_selection_report.txt"
    monkeypatch.setattr(model_train, "REPORT_PATH", str(report_path))
    results = [
        {
            "model_name": "LogisticRegression",
            "best_params": {"C": 1.0},
            "cv_accuracy": 0.81,
            "cv_precision": 0.82,
            "cv_recall": 0.83,
            "cv_roc_auc": 0.84,
            "accuracy": 0.85,
            "precision": 0.86,
            "recall": 0.87,
            "roc_auc": 0.88,
            "confusion_matrix": np.array([[3, 1], [0, 4]]),
        }
    ]

    written = model_train.save_report(results, "LogisticRegression", {"C": 1.0}, 0.88)

    assert written == report_path
    report_text = report_path.read_text(encoding="utf-8")
    assert "Model Selection and Tuning Report" in report_text
    assert "Selected best model: LogisticRegression" in report_text


def test_save_metrics_comparison_chart_creates_png(tmp_path, monkeypatch):
    chart_path = tmp_path / "reports" / "metrics.png"
    monkeypatch.setattr(model_train, "METRICS_CHART_PATH", str(chart_path))
    results = [
        {
            "model_name": "LogisticRegression",
            "accuracy": 0.85,
            "precision": 0.86,
            "recall": 0.87,
            "roc_auc": 0.88,
        },
        {
            "model_name": "RandomForest",
            "accuracy": 0.80,
            "precision": 0.81,
            "recall": 0.82,
            "roc_auc": 0.83,
        },
    ]

    written = model_train.save_metrics_comparison_chart(results)

    assert written == chart_path
    assert chart_path.exists()
    assert chart_path.stat().st_size > 0
