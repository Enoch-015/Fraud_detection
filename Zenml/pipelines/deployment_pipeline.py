import numpy as np
import pandas as pd
import json
import sys
import os
import logging
import time
import requests
import psycopg2
from zenml.client import Client
from zenml import pipeline, step
from zenml.services.service import ServiceConfig
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService, MLFlowDeploymentConfig
from zenml.integrations.mlflow import MLFLOW
from pydantic import BaseModel
import mlflow
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from steps.clean_data import clean_df
from steps.ingest_data import ingest_df
from steps.model_train import train_model
from zenml import step, get_step_context
from datetime import datetime

# Define Docker settings for ZenML
docker_settings = DockerSettings(required_integrations=[MLFLOW])

# Check the active stack for an MLflow model deployer component
client = Client()
mlflow_model_deployer_component = client.active_stack.model_deployer
if not isinstance(mlflow_model_deployer_component, MLFlowModelDeployer):
    raise RuntimeError("The active stack does not have an MLflow model deployer component.")

# Set the model deployer to the same URI
client = Client()
stack = client.active_stack
deployer = stack.model_deployer

# Print the deployer's configuration
print(deployer.config)

# Configuration for deployment trigger
class DeploymentTriggerConfig(BaseModel):
    min_accuracy: float = 0.8

DEFAULT_SERVICE_START_STOP_TIMEOUT = 60  # Timeout for starting/stopping services in seconds

@step
def deployment_trigger(
    accuracy: float,
) -> bool:
    min_accuracy: float = 0.8
    """Simple model deployment trigger based on accuracy."""
    deploy_decision = accuracy >= min_accuracy
    return deploy_decision

@step
def enhanced_model_deployer(
    deploy_decision: bool,
    model_uri: str,
) -> None:
    """Deploy the model using MLFlow if the deploy decision is True,
    after deleting any existing deployments using a bash script."""
    if not deploy_decision:
        print("Model did not meet deployment criteria. Skipping deployment.")
        return

    # Call the bash script to delete existing deployed models.
    print("Deleting existing deployed models using bash script...")
    delete_deployed_services()  # This function calls the bash script via subprocess

    # Retrieve the active model deployer
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    if not model_deployer:
        raise RuntimeError("No active MLflow model deployer found.")

    try:
        # Retrieve context inside the step without requiring it as an input
        context = get_step_context()
        unique_name = f"{context.pipeline.name}_{context.step_run.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_name = f"my_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow_deployment_config = MLFlowDeploymentConfig(
            name=unique_name,
            pipeline_name=context.pipeline.name,
            pipeline_step_name=context.step_run.name,
            model_uri=model_uri,
            model_name=model_name,  # Replace with your registered model's name if necessary
            workers=1,
            mlserver=False,
            timeout=DEFAULT_SERVICE_START_STOP_TIMEOUT,
        )

        # Deploy the new model
        service = model_deployer.deploy_model(
            config=mlflow_deployment_config,
            replace=True,
            timeout=300,
            service_type=MLFlowDeploymentService.SERVICE_TYPE,
        )
        service.start(timeout=60)
        if service.prediction_url:
            print(f"Model deployed at {service.prediction_url}")
        else:
            print("Model deployed, but prediction URL is not available.")
    except Exception as e:
        print(f"Error during model deployment: {e}")

def delete_deployed_services():
    """Delete all currently deployed services using the ZenML client API."""
    client = Client()
    services = client.list_services()
    if not services:
        print("No deployed services found.")
        return
    for service in services:
        print(f"Deleting service: UUID: {service.id}, Status: {service.status}")
        client.delete_service(service.id)
    print("Deletion of all deployed services is complete.")

class EnhancedModelDeployerParameters(BaseModel):
    model_uri: str
    deploy_decision: bool

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def core_ml_pipeline():
    df = ingest_df()
    X_train, y_train  = clean_df(df)
    model_uri = train_model(X_train,y_train)
    deploy_decision = True
    enhanced_model_deployer(model_uri=model_uri, deploy_decision=deploy_decision)

# Run the pipeline
if __name__ == "__main__":
    core_ml_pipeline()