import logging
import pandas as pd
from zenml import step

import mlflow
from zenml.client import Client
import numpy as np
from mlflow.models.signature import infer_signature

from src.model_dev import XGBClassifierModel
from sklearn.base import RegressorMixin
# from steps.config import ModelNameConfig
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
experiment_tracker = Client().active_stack.experiment_tracker

XGBClassifier = "XGBClassifier"

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
) -> str:
    if X_train.empty or y_train.empty:
        raise ValueError("Training or testing data is empty. Please check your input data.")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("Mismatch between features and target samples.")

    try:
        # Make sure we have a clean MLflow state
        mlflow.end_run()
        
        # Start a new MLflow run
        mlflow.start_run(nested=True)
        
        # Turn on autologging but be cautious it might capture too much
        mlflow.sklearn.autolog(log_models=False)  # Don't auto-log the model
        
        # Create and train the model
        logging.info("Creating XGBoost model")
        model_instance = XGBClassifierModel()
        trained_model = model_instance.train(X_train, y_train)
        
        # Generate a small sample for signature inference
        sample_input = X_train.iloc[:5].copy()  # Use a small sample of your data
        
        # Generate predictions for signature
        try:
            sample_output = trained_model.predict(sample_input)
            # Create a model signature
            signature = infer_signature(sample_input, sample_output)
            logging.info("Model signature inferred successfully")
        except Exception as sig_error:
            logging.warning(f"Failed to infer signature: {sig_error}")
            signature = None
        
        # Log custom metrics if needed
        try:
            # Log some simple metrics about the data
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_samples", X_train.shape[0])
            mlflow.log_param("model_type", "XGBoost")
            
            # You could add more custom metrics here if needed
        except Exception as metric_error:
            logging.warning(f"Error logging metrics: {metric_error}")
        
        # Log the model with explicit artifact path and proper formats
        logging.info("Logging model to MLflow")
        
        # Use specific arguments to ensure proper logging
        mlflow.sklearn.log_model(
            sk_model=trained_model,
            artifact_path="model_dir",
            signature=signature,
            input_example=sample_input,
            registered_model_name="fraud_detection_model"
        )
        
        # Get the model URI for return
        model_uri = mlflow.get_artifact_uri("model_dir")
        logging.info(f"Model logged to MLflow at: {model_uri}")
        
        return model_uri
    
    except Exception as e:
        logging.error(f"Error in training Model: {e}")
        raise e
    
    finally:
        # Always end the MLflow run
        try:
            mlflow.end_run()
        except Exception as end_run_error:
            logging.warning(f"Error ending MLflow run: {end_run_error}")