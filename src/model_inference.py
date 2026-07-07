import argparse
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient


DEFAULT_MODEL_PATH = "models/best_model.pkl"
DEFAULT_CLEANED_DATA_PATH = "data/processed/heart_disease_cleaned.csv"
DEFAULT_INFERENCE_DATA_PATH = "data/processed/heart_disease_inference.csv"
DEFAULT_OUTPUT_PATH = "outputs/predict/predictions.csv"
DEFAULT_MLFLOW_DIR = "outputs/mlflow"
DEFAULT_EXPERIMENT_NAME = "heart_disease_inference"
TARGET_COLUMN = "target"


def resolve_path(path: str) -> Path:
    return Path(path).resolve()


def configure_mlflow(mlflow_dir: str, experiment_name: str) -> str:
    mlflow_root = resolve_path(mlflow_dir)
    db_path = mlflow_root / "ML_Inference_mlflow.db"
    artifact_root = mlflow_root / "ML_Inference_artifacts"

    mlflow_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    artifact_location = artifact_root.as_uri()

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = client.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    print(f"MLflow tracking database: {db_path}")
    print(f"MLflow artifact folder: {artifact_root}")

    return experiment_id


def load_model(model_path: str):
    file_path = Path(model_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")

    return joblib.load(file_path)


def create_inference_file(
    cleaned_data_path: str,
    inference_data_path: str,
    target_column: str,
) -> tuple[Path, pd.DataFrame]:
    cleaned_file = Path(cleaned_data_path)

    if not cleaned_file.exists():
        raise FileNotFoundError(f"Cleaned data file not found: {cleaned_file}")

    data = pd.read_csv(cleaned_file)

    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in {cleaned_file}. "
            f"Available columns: {list(data.columns)}"
        )

    inference_data = data.drop(columns=[target_column])

    inference_file = Path(inference_data_path)
    inference_file.parent.mkdir(parents=True, exist_ok=True)
    inference_data.to_csv(inference_file, index=False)

    print(f"Inference input saved to: {inference_file}")
    return inference_file, inference_data


def run_inference(
    input_path: Path,
    model_path: str,
    output_path: str,
) -> tuple[Path, pd.DataFrame]:
    data = pd.read_csv(input_path)
    model = load_model(model_path)

    predictions = model.predict(data)

    output_data = data.copy()
    output_data["prediction"] = predictions

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_data.to_csv(output_file, index=False)

    print(f"Input rows: {len(data)}")
    print(f"Predictions saved to: {output_file}")

    return output_file, output_data


def parse_args():
    parser = argparse.ArgumentParser(description="Run model inference with MLflow tracking.")

    parser.add_argument(
        "--cleaned-data",
        default=DEFAULT_CLEANED_DATA_PATH,
        help=f"Path to cleaned CSV file containing the target column. Default: {DEFAULT_CLEANED_DATA_PATH}",
    )
    parser.add_argument(
        "--inference-data",
        default=DEFAULT_INFERENCE_DATA_PATH,
        help=f"Path to save the target-free inference CSV. Default: {DEFAULT_INFERENCE_DATA_PATH}",
    )
    parser.add_argument(
        "--target-column",
        default=TARGET_COLUMN,
        help=f"Target column to drop before inference. Default: {TARGET_COLUMN}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to trained model pickle file. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to save predictions CSV. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--mlflow-dir",
        default=DEFAULT_MLFLOW_DIR,
        help=f"Folder for MLflow database and artifacts. Default: {DEFAULT_MLFLOW_DIR}",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help=f"MLflow experiment name. Default: {DEFAULT_EXPERIMENT_NAME}",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    configure_mlflow(
        mlflow_dir=args.mlflow_dir,
        experiment_name=args.experiment_name,
    )

    with mlflow.start_run(run_name="model_inference"):
        mlflow.log_params(
            {
                "cleaned_data_path": args.cleaned_data,
                "inference_data_path": args.inference_data,
                "target_column": args.target_column,
                "model_path": args.model,
                "output_path": args.output,
                "mlflow_dir": args.mlflow_dir,
            }
        )

        inference_file, inference_data = create_inference_file(
            cleaned_data_path=args.cleaned_data,
            inference_data_path=args.inference_data,
            target_column=args.target_column,
        )

        output_file, output_data = run_inference(
            input_path=inference_file,
            model_path=args.model,
            output_path=args.output,
        )

        mlflow.log_metrics(
            {
                "inference_rows": len(inference_data),
                "inference_columns": len(inference_data.columns),
                "prediction_rows": len(output_data),
            }
        )

        mlflow.log_artifact(str(inference_file), artifact_path="inference_data")
        mlflow.log_artifact(str(output_file), artifact_path="predictions")

        run = mlflow.active_run()
        print(f"MLflow experiment: {args.experiment_name}")
        print(f"MLflow run_id: {run.info.run_id}")


if __name__ == "__main__":
    main()
