import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


class DataStrategy(ABC):
    """
    Abstaract Class for determining strategy for handling data
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
    


class DataDivideStrategy(DataStrategy):
    """ 
    Strategy dividing data into the train and test 
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """ 
        Divide data into the train and test
        """
        try:
            X_train = data.drop(["Class"], axis=1)
            y_train = data["Class"]
            return X_train,y_train
        except Exception as e:
            logging.error("Error dividing data: {}".format(e))
            raise e

class DataCleaning:
    """
    Class for cleaning data which processes the data and divides it into train and test data
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy
        
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """ 
        Handle data
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e