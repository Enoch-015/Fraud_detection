from zenml.steps.base_parameters import BaseParameters

class ModelNameConfig(BaseParameters):
    """Model Configs"""
    model_name: str = "XGBClassifier"