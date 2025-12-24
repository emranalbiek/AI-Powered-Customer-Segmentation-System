import logging
import os
import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler

# Create Class for Feature Engineering
class FeatureEngineering():
    def apply_log_transformation(self, df:pd.DataFrame) -> pd.DataFrame:
        """Applies Log Transformation on the dataframe"""
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.to_list()
        logging.info(f"Applying log transformation to features: {numeric_cols}")
        transformed_df = df.copy()
        
        for feature in numeric_cols:
            transformed_df[feature] = np.log1p(
                transformed_df[feature]
        ) # log1p handles log(0) by calculating log(1+x)
            logging.info(f"Applied log1p to feature: {feature}")
        logging.info("Log Transformation Completed")
        return transformed_df
    
    def apply_outliers_handling(self, df:pd.DataFrame, contamination=0.01, method:str='cap') -> pd.DataFrame:
        """Applies Outliers Handling on train set"""
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_df = df[numeric_cols].fillna(0)
        
        # Initialize and fit Isolation Forest
        model = IsolationForest(
            contamination=contamination, random_state=42
        )
        preds = model.fit_predict(numeric_df)
        
        # Convert predictions to boolean DataFrame
        outliers = pd.DataFrame(preds == -1, columns=["is_outlier"], index=df.index)
        
        logging.info(f"{outliers['is_outlier'].sum()} detected.")
        
        if method == "remove":
            logging.info("Removing outliers from the dataset.")
            handled_df = df.copy()
            handled_df = df[(~outliers).all(axis=1)]
        
        elif method == "cap":
            logging.info("Capping outliers in the dataset.")
            handled_df = df.copy()
            non_numeric_cols = df.select_dtypes(exclude=np.number).columns
            for col in numeric_cols:
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                handled_df[col] = df[col].clip(lower=lower, upper=upper)
            
            for col in non_numeric_cols:
                handled_df[col] = df[col]
        
        else:
            logging.warning(f"Unknown method '{method}'. No outlier handling performed.")
            return df
        
        logging.info("Outlier handling completed.")
        return handled_df
    
    def apply_feature_scaling(self, df:pd.DataFrame, strategy:str) -> pd.DataFrame:
        """Applies Feature Scaling on train and validation sets"""
        feature_columns = [col for col in df.columns if col != 'CustomerID']
        
        df = df.drop(columns=['CustomerID'])
        
        if strategy == 'standard_scaler':
            scaler = StandardScaler()
            
            scaled_data = scaler.fit_transform(df)
        
        elif strategy == 'robust_scaler':
            scaler = RobustScaler()
            
            scaled_data = scaler.fit_transform(df)
        
        else:
            logging.warning(f"Unknown method '{strategy}'. No feature scaling performed.")
            return df
        
        # Save the scaler
        # Ensure the artifacts directory exists
        if not os.path.exists(r'.\artifacts'):
            os.makedirs(r'.\artifacts')
        
        # Ensure the pickle file is not existing
        file_path = os.path.join(r'.\artifacts', f'{strategy}.pkl')
        if not os.path.exists(file_path):
            with open(file_path, 'wb') as f:
                pickle.dump(scaler, f)
        
        scaled_df = pd.DataFrame(scaled_data, columns=feature_columns)
        
        logging.info('Feature Scaling Completed')
        return scaled_df