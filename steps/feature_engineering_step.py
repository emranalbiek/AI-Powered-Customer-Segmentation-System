import logging
import pandas as pd

from src.feature_engineering import FeatureEngineering
from src.extract_features import FeaturesExtraction

def feature_engineering_step(df:pd.DataFrame):
    try:
        feature_engineering = FeatureEngineering()
        feature_extraction = FeaturesExtraction()
        
        extracted_df = feature_extraction.extract(df)
        transformed_df = feature_engineering.apply_log_transformation(extracted_df)
        scaled_df = feature_engineering.apply_feature_scaling(transformed_df, strategy='robust_scaler')
        
        logging.info('Feature Engineering Completed Successfully')
        return scaled_df, extracted_df
    except Exception as e:
        logging.error(f'Error in feature engineering: {e}')
        raise e