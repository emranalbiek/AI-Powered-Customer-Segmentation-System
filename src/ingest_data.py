import os
import zipfile
from abc import ABC, abstractmethod

import pandas as pd


# Define an abstract class for Data Ingestor
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Abstract method to ingest data from a given file."""
        pass


# Implement a concrete class for ZIP Ingestion
class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Extracts a .zip file and returns the content as a pandas DataFrame."""
        # Ensure the file is a .zip
        if not file_path.endswith(".zip"):
            raise ValueError("The provided file is not a .zip file.")
        
        # Extract the zip file
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall("extracted_data")
        
        # Find the extracted XLSX file (assuming there is one XLSX file inside the zip)
        extracted_files = os.listdir("extracted_data")
        excel_files = [f for f in extracted_files if f.endswith(".xlsx")]
        
        if len(excel_files) == 0:
            raise FileNotFoundError("No XLSX file found in the extracted data.")
        if len(excel_files) > 1:
            raise ValueError("Multiple XLSX files found. Please specify which one to use.")
        
        # Read the XLSX into a DataFrame
        excel_file_path = os.path.join("extracted_data", excel_files[0])
        df = pd.read_excel(excel_file_path)
        
        # Return the DataFrame
        return df


# Implement a Factory to create DataIngestors
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """Returns the appropriate DataIngestor based on file extension."""
        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file extension: {file_extension}")