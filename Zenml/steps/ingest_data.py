import logging
import pandas as pd
from zenml import step
from sqlalchemy import create_engine
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

 
@step
def ingest_df() -> pd.DataFrame:
    """
    Fetch data from the database.
    
    Args:
        db_url (str): Database connection string.
        query (str): SQL query to fetch data.
    
    Returns:
        pd.DataFrame: DataFrame containing the data.
    """
    try:
        # Create a connection to the database
        db_url = "postgresql://floo:Agents1234@floo3.postgres.database.azure.com:5432/postgres"
        query = "SELECT * FROM fraud_dec_training"
        engine = create_engine(db_url)
        logging.info("Connecting to the database...")
        with engine.connect() as connection:
            logging.info("Executing query...")
            df = pd.read_sql(query, connection)
        logging.info("Data fetched successfully.")
        return df
    except Exception as e:
        logging.error(f"Error fetching data from the database: {e}")
        raise