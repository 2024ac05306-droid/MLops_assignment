"""Configuration management for model training."""
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseSettings

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings."""
    
    # Model Configuration
    model_type: str = os.getenv("MODEL_TYPE", "neural_network")
    batch_size: int = int(os.getenv("BATCH_SIZE", 32))
    epochs: int = int(os.getenv("EPOCHS", 10))
    learning_rate: float = float(os.getenv("LEARNING_RATE", 0.001))
    random_seed: int = int(os.getenv("RANDOM_SEED", 42))
    
    # Data Configuration
    data_path: Path = Path(os.getenv("DATA_PATH", "./data/raw"))
    processed_data_path: Path = Path(os.getenv("PROCESSED_DATA_PATH", "./data/processed"))
    test_size: float = float(os.getenv("TEST_SIZE", 0.2))
    
    # Output Configuration
    model_output_path: Path = Path(os.getenv("MODEL_OUTPUT_PATH", "./models"))
    log_dir: Path = Path(os.getenv("LOG_DIR", "./logs"))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "./outputs"))
    
    # MLflow Configuration
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops-assignment")
    
    # TensorBoard
    tensorboard_log_dir: Path = Path(os.getenv("TENSORBOARD_LOG_DIR", "./logs/tensorboard"))
    
    # Environment
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Create settings instance
settings = Settings()

# Create necessary directories
for directory in [
    settings.data_path,
    settings.processed_data_path,
    settings.model_output_path,
    settings.log_dir,
    settings.output_dir,
    settings.tensorboard_log_dir,
]:
    directory.mkdir(parents=True, exist_ok=True)
