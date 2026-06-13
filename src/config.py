"""Configuration management for Heart Disease Classification MLOps Project."""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application Settings"""

    # Model Configuration
    model_types: str = os.getenv(
        "MODEL_TYPES",
        "LogisticRegression,RandomForest"
    )

    random_seed: int = int(
        os.getenv("RANDOM_SEED", 42)
    )

    test_size: float = float(
        os.getenv("TEST_SIZE", 0.2)
    )

    # Logistic Regression Parameters
    lr_max_iter: int = int(
        os.getenv("LR_MAX_ITER", 1000)
    )

    # Random Forest Parameters
    rf_n_estimators: int = int(
        os.getenv("RF_N_ESTIMATORS", 100)
    )

    rf_max_depth: int = int(
        os.getenv("RF_MAX_DEPTH", 10)
    )

    # Data Configuration
    data_path: Path = Path(
        os.getenv(
            "DATA_PATH",
            "./data/raw"
        )
    )

    processed_data_path: Path = Path(
        os.getenv(
            "PROCESSED_DATA_PATH",
            "./data/processed"
        )
    )
    test_size: float = float(os.getenv("TEST_SIZE", 0.2))

    # Output Configuration
    model_output_path: Path = Path(
        os.getenv(
            "MODEL_OUTPUT_PATH",
            "./models"
        )
    )

    log_dir: Path = Path(
        os.getenv(
            "LOG_DIR",
            "./logs"
        )
    )

    output_dir: Path = Path(
        os.getenv(
            "OUTPUT_DIR",
            "./outputs"
        )
    )

    # MLflow Configuration
    mlflow_tracking_uri: str = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000"
    )

    mlflow_experiment_name: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "HeartDiseaseClassification"
    )

    # Environment
    environment: str = os.getenv(
        "ENVIRONMENT",
        "development"
    )

    debug: bool = os.getenv(
        "DEBUG",
        "True"
    ).lower() == "true"

    class Config:
        env_file = ".env"
        case_sensitive = False


# Create settings instance
settings = Settings()

# Create required directories
for directory in [
    settings.data_path,
    settings.processed_data_path,
    settings.model_output_path,
    settings.log_dir,
    settings.output_dir,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )
