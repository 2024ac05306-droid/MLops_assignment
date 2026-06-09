"""Main training script for ML model - Heart Disease Classification."""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
import joblib
import json
import mlflow
import mlflow.sklearn
from urllib.request import urlretrieve
import warnings

from config import settings

warnings.filterwarnings('ignore')

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


def download_dataset():
    """Download Heart Disease UCI dataset."""
    logger.info("Downloading Heart Disease UCI dataset...")
    
    data_path = settings.data_path / 'heart_disease.csv'
    
    if data_path.exists():
        logger.info(f"Dataset already exists at {data_path}")
        return data_path
    
    url = "https://raw.githubusercontent.com/datasets/uci-ml-datasets/master/heart-disease/heart-disease.csv"
    
    try:
        urlretrieve(url, data_path)
        logger.info(f"Dataset downloaded successfully to {data_path}")
    except Exception as e:
        logger.warning(f"Could not download from URL: {e}")
        logger.info("Using local dataset or alternative source...")
        # Fallback: assume data is already in data_path
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    return data_path


def load_and_clean_data():
    """Load and clean the Heart Disease dataset."""
    logger.info("Loading and cleaning dataset...")
    
    data_path = download_dataset()
    df = pd.read_csv(data_path)
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Dataset columns: {df.columns.tolist()}")
    
    # Handle missing values
    logger.info("Handling missing values...")
    initial_missing = df.isnull().sum().sum()
    logger.info(f"Initial missing values: {initial_missing}")
    
    # Fill missing values with median for numerical columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    final_missing = df.isnull().sum().sum()
    logger.info(f"Final missing values: {final_missing}")
    
    # Remove duplicates
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
    
    # Log class distribution
    if 'target' in df.columns:
        logger.info("Target variable distribution:")
        logger.info(df['target'].value_counts())
    
    return df


def preprocess_and_split_data(df):
    """Preprocess and split the dataset."""
    logger.info("Preprocessing and splitting dataset...")
    
    # Separate features and target
    # Assuming 'target' is the target column (0: no disease, 1: disease present)
    if 'target' in df.columns:
        X = df.drop('target', axis=1)
        y = df['target']
    else:
        # If target column has different name, use last column
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        logger.warning("Using last column as target variable")
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    
    # Handle categorical features
    categorical_columns = X.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        logger.info(f"Encoding categorical features: {categorical_columns.tolist()}")
        X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=settings.test_size,
        random_state=settings.random_seed,
        stratify=y  # Stratified split for imbalanced datasets
    )
    
    logger.info(f"Training set size: {X_train.shape}")
    logger.info(f"Test set size: {X_test.shape}")
    logger.info(f"Training set class distribution:\n{pd.Series(y_train).value_counts()}")
    logger.info(f"Test set class distribution:\n{pd.Series(y_test).value_counts()}")
    
    return X_train, X_test, y_train, y_test, X.columns


def scale_data(X_train, X_test):
    """Scale features using StandardScaler."""
    logger.info("Scaling features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    scaler_path = settings.model_output_path / 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    
    return X_train_scaled, X_test_scaled, scaler


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model."""
    logger.info("Training Logistic Regression model...")
    
    model = LogisticRegression(
        random_state=settings.random_seed,
        max_iter=1000,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    logger.info("Logistic Regression training completed")
    
    return model


def train_random_forest(X_train, y_train):
    """Train Random Forest model."""
    logger.info("Training Random Forest model...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=settings.random_seed,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )
    
    model.fit(X_train, y_train)
    logger.info("Random Forest training completed")
    
    return model


def cross_validate_model(model, X_train, y_train, model_name):
    """Perform cross-validation on the model."""
    logger.info(f"Performing 5-fold cross-validation for {model_name}...")
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    cv_results = cross_validate(
        model, X_train, y_train,
        cv=5,
        scoring=scoring,
        return_train_score=True
    )
    
    # Log cross-validation results
    for metric in scoring.keys():
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        logger.info(f"{model_name} CV {metric}:")
        logger.info(f"  Train: {train_scores.mean():.4f} (+/- {train_scores.std():.4f})")
        logger.info(f"  Test:  {test_scores.mean():.4f} (+/- {test_scores.std():.4f})")
    
    return cv_results


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance on test set."""
    logger.info(f"Evaluating {model_name} on test set...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    logger.info(f"\n{model_name} Test Metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    # Log confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    report = classification_report(y_test, y_pred)
    logger.info(f"\nClassification Report:\n{report}")
    
    return metrics, y_pred, y_pred_proba


def save_model(model, model_path, model_name):
    """Save trained model."""
    joblib.dump(model, model_path)
    logger.info(f"{model_name} saved to {model_path}")
    return model_path


def main():
    """Main training pipeline."""
    try:
        logger.info("="*70)
        logger.info("Starting Heart Disease Classification Training Pipeline")
        logger.info("="*70)
        
        # Set MLflow experiment
        mlflow.set_experiment(settings.mlflow_experiment_name)
        
        with mlflow.start_run(run_name="heart_disease_classification"):
            # Load and preprocess data
            df = load_and_clean_data()
            X_train, X_test, y_train, y_test, feature_names = preprocess_and_split_data(df)
            
            # Scale data
            X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
            
            # Log data statistics
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("n_features", X_train_scaled.shape[1])
            mlflow.log_param("random_seed", settings.random_seed)
            
            # Initialize results storage
            all_metrics = {}
            all_cv_results = {}
            
            # ============ Train Logistic Regression ============
            logger.info("\n" + "="*70)
            logger.info("LOGISTIC REGRESSION MODEL")
            logger.info("="*70)
            
            lr_model = train_logistic_regression(X_train_scaled, y_train)
            lr_cv_results = cross_validate_model(lr_model, X_train_scaled, y_train, "Logistic Regression")
            lr_metrics, lr_pred, lr_pred_proba = evaluate_model(
                lr_model, X_test_scaled, y_test, "Logistic Regression"
            )
            
            all_metrics['logistic_regression'] = lr_metrics
            all_cv_results['logistic_regression'] = lr_cv_results
            
            # Log Logistic Regression metrics to MLflow
            for metric_name, metric_value in lr_metrics.items():
                mlflow.log_metric(f"lr_{metric_name}", metric_value)
            
            # ============ Train Random Forest ============
            logger.info("\n" + "="*70)
            logger.info("RANDOM FOREST MODEL")
            logger.info("="*70)
            
            rf_model = train_random_forest(X_train_scaled, y_train)
            rf_cv_results = cross_validate_model(rf_model, X_train_scaled, y_train, "Random Forest")
            rf_metrics, rf_pred, rf_pred_proba = evaluate_model(
                rf_model, X_test_scaled, y_test, "Random Forest"
            )
            
            all_metrics['random_forest'] = rf_metrics
            all_cv_results['random_forest'] = rf_cv_results
            
            # Log Random Forest metrics to MLflow
            for metric_name, metric_value in rf_metrics.items():
                mlflow.log_metric(f"rf_{metric_name}", metric_value)
            
            # ============ Model Comparison ============
            logger.info("\n" + "="*70)
            logger.info("MODEL COMPARISON")
            logger.info("="*70)
            
            logger.info("\nLogistic Regression Metrics:")
            for metric_name, metric_value in lr_metrics.items():
                logger.info(f"  {metric_name}: {metric_value:.4f}")
            
            logger.info("\nRandom Forest Metrics:")
            for metric_name, metric_value in rf_metrics.items():
                logger.info(f"  {metric_name}: {metric_value:.4f}")
            
            # Determine best model
            best_model_name = "Random Forest" if rf_metrics['f1'] > lr_metrics['f1'] else "Logistic Regression"
            best_model = rf_model if rf_metrics['f1'] > lr_metrics['f1'] else lr_model
            logger.info(f"\nBest Model: {best_model_name} (F1-Score: {max(rf_metrics['f1'], lr_metrics['f1']):.4f})")
            
            # ============ Save Models ============
            logger.info("\n" + "="*70)
            logger.info("SAVING MODELS")
            logger.info("="*70)
            
            lr_model_path = settings.model_output_path / 'logistic_regression_model.pkl'
            save_model(lr_model, lr_model_path, "Logistic Regression")
            
            rf_model_path = settings.model_output_path / 'random_forest_model.pkl'
            save_model(rf_model, rf_model_path, "Random Forest")
            
            best_model_path = settings.model_output_path / 'best_model.pkl'
            save_model(best_model, best_model_path, f"Best Model ({best_model_name})")
            
            # Log model artifacts to MLflow
            mlflow.sklearn.log_model(lr_model, "logistic_regression_model")
            mlflow.sklearn.log_model(rf_model, "random_forest_model")
            mlflow.sklearn.log_model(best_model, "best_model")
            
            # ============ Save Metrics and Results ============
            logger.info("\n" + "="*70)
            logger.info("SAVING RESULTS")
            logger.info("="*70)
            
            # Save metrics to JSON
            metrics_path = settings.output_dir / 'model_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(all_metrics, f, indent=4)
            logger.info(f"Metrics saved to {metrics_path}")
            
            # Save detailed results
            results = {
                'best_model': best_model_name,
                'logistic_regression': all_metrics['logistic_regression'],
                'random_forest': all_metrics['random_forest'],
                'feature_names': feature_names.tolist(),
                'n_features': len(feature_names),
                'random_seed': settings.random_seed
            }
            
            results_path = settings.output_dir / 'training_results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            logger.info(f"Training results saved to {results_path}")
            
            # Log artifacts
            mlflow.log_artifact(str(metrics_path))
            mlflow.log_artifact(str(results_path))
            
            logger.info("\n" + "="*70)
            logger.info("Training Pipeline Completed Successfully!")
            logger.info("="*70)
            
            return 0
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        mlflow.log_param("error", str(e))
        return 1


if __name__ == "__main__":
    exit(main())
