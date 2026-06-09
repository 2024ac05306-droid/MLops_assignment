"""Main training script for ML model."""
import logging
import numpy as np
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.log_dir / 'training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_data():
    """Load sample dataset."""
    logger.info("Loading dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=settings.test_size,
        random_state=settings.random_seed
    )
    
    logger.info(f"Training set size: {X_train.shape}")
    logger.info(f"Test set size: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test):
    """Preprocess data."""
    logger.info("Preprocessing data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, settings.model_output_path / 'scaler.pkl')
    logger.info("Scaler saved")
    
    return X_train_scaled, X_test_scaled


def train_model(X_train, y_train):
    """Train the model."""
    logger.info(f"Training {settings.model_type} model...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=settings.random_seed,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    logger.info("Evaluating model...")
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    return metrics


def save_model(model):
    """Save trained model."""
    model_path = settings.model_output_path / 'model.pkl'
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    return model_path


def main():
    """Main training pipeline."""
    try:
        logger.info("="*50)
        logger.info("Starting Model Training Pipeline")
        logger.info("="*50)
        
        # Load data
        X_train, X_test, y_train, y_test = load_data()
        
        # Preprocess data
        X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)
        
        # Train model
        model = train_model(X_train_scaled, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test_scaled, y_test)
        
        # Save model
        model_path = save_model(model)
        
        logger.info("="*50)
        logger.info("Training Pipeline Completed Successfully")
        logger.info("="*50)
        
        # Save metrics
        with open(settings.output_dir / 'metrics.txt', 'w') as f:
            for metric_name, metric_value in metrics.items():
                f.write(f"{metric_name}: {metric_value:.4f}\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
