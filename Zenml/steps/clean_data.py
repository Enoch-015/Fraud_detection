import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataDivideStrategy
from typing_extensions import Annotated
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.Series, "y_train"],
]:
    
    """ 
    Cleans the data and divides it into train and test
    
    Args:
        df: Raw Data
    Returns:
        X_train = Training data
        y_train: Training Labels
    """
    logging.info("Cleaning data")
    try:
        if df is None:
            raise ValueError("Input DataFrame is None")
        

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(df,divide_strategy)
        X_train,y_train = data_cleaning.handle_data()
        if X_train is None or y_train is None :
            raise ValueError("One or more of the returned datasets is None")
        
        
        logging.info("Data Cleaning Completed")
        return X_train,y_train
    except Exception as e:
        logging.error(f"Error during data cleaning: {e}")
        raise e