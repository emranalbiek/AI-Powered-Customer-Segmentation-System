import logging
import pandas as pd

from src.evaluate_model import InternalEvaluation, VisualEvaluation, BusinessEvaluation
from steps.configs import ModelConfig

config = ModelConfig()

def model_evaluation_step(engineered_df:pd.DataFrame, original_df:pd.DataFrame, labels):
    try:
        silhouette_score, davies_bouldin_score, calinski_harabasz_score = InternalEvaluation().evaluate_model(engineered_df, labels)
        segment_analysis = BusinessEvaluation().evaluate_model(original_df, labels)
        VisualEvaluation().evaluate_model(engineered_df, labels)
        
        logging.info('Model Evaluation Completed Successfully')
        return silhouette_score, davies_bouldin_score, calinski_harabasz_score, segment_analysis
    except Exception as e:
        logging.error(f'Error in Model Evaluation: {e}')
        raise e