import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np


with mlflow.start_run():
    model = mlflow.sklearn.load_model("runs:/f1e3c5b0a2d14e4a8b6c9e1f2b3c4d5e/model")
    mlflow.sklearn.log_model(model, "model")
    model.fit(X_train, y_train)
    
