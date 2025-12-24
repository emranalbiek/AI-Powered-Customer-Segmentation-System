import logging
import pandas as pd
from src.feature_engineering import FeatureEngineering
import pickle


def create_test_set():
    try:
        data = {
            'CustomerID': ['1000', '1001', '1002', '1003', '1004'],
            'Recency': [11, 68, 120, 230, 500],
            'Frequency': [15, 8, 5, 3, 2],
            'Monetary': [12040.0, 4589.0, 1790.0, 990.0, 700.0],
            'Avg_Basket_Size': [2000.0, 1090.98, 400.65, 323.0, 255.0],
            'CLV': [7150.82, 3290.0, 2323.0, 1008.54, 319.56]
        }
        
        df = pd.DataFrame(data=data)
        
        feature_engineering = FeatureEngineering()
        transformed_df = feature_engineering.apply_log_transformation(df)
        
        transformed_df = transformed_df.drop(columns=['CustomerID'])
        
        scaler_path = r'C:\Users\mo\End to End Customers Segmentation System Project\artifacts\robust_scaler.pkl'
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
        
        scaled_df = scaler.transform(transformed_df)
        
        logging.info("Test Data Creation Completed")
        return scaled_df
    except Exception as e:
        logging.error(f"Error in test data creation: {e}")
        raise e