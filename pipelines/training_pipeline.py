import logging

from steps.data_ingestion import data_ingestion_step
from steps.data_cleaning import data_cleaning_step
from steps.feature_engineering_step import feature_engineering_step
from steps.model_building import model_building_step
from steps.configs import ModelConfig
from steps.model_evaluation import model_evaluation_step
from tests.get_test_data import create_test_set

config = ModelConfig()

def training_pipeline():
    # Ingest the data
    logging.info('[data ingestion]')
    df = data_ingestion_step(config.file_path)
    # Clean the data
    logging.info('[data cleaning]')
    clean_df = data_cleaning_step(df)
    # Apply feature engineering techniques
    logging.info('[feature engineering]')
    engineered_df, original_df = feature_engineering_step(clean_df)
    # Train the model
    logging.info('[model training]')
    model, predictions = model_building_step(engineered_df, config)
    # Get the test set and predictions
    engineered_test_df = create_test_set()
    test_preds = model.predict(engineered_test_df)
    # Evaluate the model
    logging.info('[model evaluation]')
    silhouette_score, davies_bouldin_score, calinski_harabasz_score, segment_analysis = model_evaluation_step(engineered_df, original_df, predictions)
    # Test data predictions
    logging.info('[Test data predictions]')
    print(f"Test Predictions: {test_preds}")
    
    logging.info("Training Pipeline Completed Successfully")
    return model