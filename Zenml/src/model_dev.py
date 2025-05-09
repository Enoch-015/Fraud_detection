import logging
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve, roc_auc_score, classification_report, confusion_matrix, average_precision_score, auc
from sklearn.base import BaseEstimator, ClassifierMixin


class Model(ABC):
    """ 
    Abstract class for all models
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """ 
        Trains the model
        
        Args:
            X_train: Training data
            y_train: Training label
        Returns:
            Trained model
        """
        pass


class XGBClassifierModel(Model):
    """ 
    XGBOOST Classifier Model wrapper to ensure compatibility with MLflow
    """
    def train(self, X_train, y_train, **kwargs):
        """ 
        Trains the model
        Args:
            X_train: Training data
            y_train: Training label
        Return:
            Trained model instance that inherits from sklearn BaseEstimator
        """ 
        try:
            logging.info("Model Training has started")
            
            # First, ensure y_train is in the right format - numpy array of integers
            y_train_array = np.asarray(y_train)
            
            # Check if y_train is not a simple array (it might be a pandas Series with complex structure)
            if y_train_array.ndim > 1 or y_train_array.dtype.kind not in 'iuf':
                logging.info("Converting y_train to appropriate format for bincount")
                # Try to extract values if it's a pandas Series
                try:
                    y_train_array = y_train.values
                except:
                    pass
                
                # Convert to integers if they aren't already
                try:
                    y_train_array = y_train_array.astype(int)
                except:
                    # If conversion fails, handle differently
                    logging.info("Could not convert to integers. Using manual class balance approach.")
                    scale_pos_weight = 1.0  # Default balanced weight
            
            # Only use bincount if we can guarantee it's a clean 1D integer array
            if y_train_array.ndim == 1 and np.issubdtype(y_train_array.dtype, np.integer):
                # Now safe to use bincount
                class_counts = np.bincount(y_train_array)
                if len(class_counts) >= 2:
                    neg, pos = class_counts[0], class_counts[1]
                    scale_pos_weight = neg / pos
                else:
                    # Handle case where there might be only one class in the data
                    logging.warning("Only one class found in training data")
                    scale_pos_weight = 1.0
            else:
                # Alternative: calculate ratio directly
                pos_count = np.sum(y_train_array == 1)
                neg_count = np.sum(y_train_array == 0)
                
                if neg_count > 0 and pos_count > 0:
                    scale_pos_weight = neg_count / pos_count
                else:
                    logging.warning("Could not determine class balance. Using default weight.")
                    scale_pos_weight = 1.0
            
            logging.info(f"Using scale_pos_weight: {scale_pos_weight}")
            
            # Apply SMOTE for oversampling
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            
            # Create an XGBoost model with updated parameters (removed use_label_encoder)
            model = XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42,
                eval_metric='logloss'  # Removed use_label_encoder as it's deprecated
            )

            # Train the model
            model.fit(X_train_resampled, y_train_resampled)
            
            logging.info("Model training completed")
            return model
            
        except Exception as e:
            logging.error(f"Error Training the model: {e}")
            raise e