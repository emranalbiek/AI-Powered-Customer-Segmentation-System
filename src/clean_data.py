import logging
import pandas as pd

# Setup logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Create Class for Data Preprocessing
class DataPreprocessing():
    def clean(self, df:pd.DataFrame) -> pd.DataFrame:
        """Cleans and Formats the data for model training
        
        Args:
            df: pandas dataframe containing customers data
        
        Returns:
            cleaned_data: pandas dataframe containing cleaned data
        """
        try:
            # Step 1:Drop Duplicates
            df = df.drop_duplicates()
            logging.info('Drop Duplicates Step Completed')
            
            # Step 2:Drop Missing Values
            df = df.dropna()
            logging.info('Drop Missing Values Step Completed')
            
            # Step 2:Change `CustomerID` Column Type
            df['CustomerID'] = df['CustomerID'].astype('str')
            df['CustomerID'] = df['CustomerID'].str.replace(r'\.0$', '', regex=True)
            logging.info('Change `CustomerID` Column Type Step Completed')
            
            # Step 3:Keep Date only in `InvoiceDate` Column
            df['InvoiceDate'] = df['InvoiceDate'].dt.normalize()
            logging.info('Keep Date only in `InvoiceDate` Column Step Completed')
            
            # Step 4:Drop Useless Columns
            useless_cols = ['Description', 'Country']
            df = df.drop(columns=useless_cols)
            logging.info('Drop Useless Columns Step Completed')
            
            logging.info('Data Preprocessing Completed Successfully')
            return df
        except Exception as e:
            logging.error(f'Error in data preprocessing: {e}')
            raise e