import logging
import pandas as pd

from src.ingest_data import DataIngestorFactory

def data_ingestion_step(file_path:str) -> pd.DataFrame:
    """Ingests data from `DataIngestorFactory` class"""
    try:
        # Define the file extension
        file_extension = '.zip'
        
        # Load the data ingestor to extract data
        data_ingestor = DataIngestorFactory().get_data_ingestor(file_extension)
        
        # Extract data as DataFrame
        df = data_ingestor.ingest(file_path)
        
        logging.info('Data Ingestion Completed')
        return df
    except Exception as e:
        logging.error(f'Error in data ingestion: {e}')
        raise e