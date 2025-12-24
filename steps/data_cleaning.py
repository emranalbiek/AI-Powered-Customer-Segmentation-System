import pandas as pd

from src.clean_data import DataPreprocessing

def data_cleaning_step(df:pd.DataFrame) -> pd.DataFrame:
    data_preprocessor = DataPreprocessing()
    clean_df = data_preprocessor.clean(df)
    return clean_df