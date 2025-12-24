import logging
import os
import pandas as pd
import pickle

from src.build_model import (
    KMeansModel, AgglomerativeModel, 
    GMModel, HyperparameterTuner)

from steps.configs import ModelConfig

def model_building_step(df:pd.DataFrame, config:ModelConfig):
    try:
        model = None
        tuner = None
        
        if config.model_name == "k_means":
            model = KMeansModel()
        elif config.model_name == "agg":
            model = AgglomerativeModel()
        elif config.model_name == "gmm":
            model = GMModel()
        else:
            raise ValueError("Model name not supported")
        
        tuner = HyperparameterTuner(model, df)
        
        if config.fine_tuning:
            best_params = tuner.optimize(n_trials=300)
            trained_model, predictions = model.train(df, **best_params)
        else:
            trained_model, predictions = model.train(df)
        
        # Save the model
        # Ensure the artifacts directory exists
        if not os.path.exists(r'.\artifacts'):
            os.makedirs(r'.\artifacts')
        
        # Ensure the pickle file is not existing
        file_path = os.path.join(r'.\artifacts', f'{config.model_name}_model.pkl')
        if not os.path.exists(file_path):
            with open(file_path, 'wb') as f:
                pickle.dump(trained_model, f)
        
        return trained_model, predictions
    except Exception as e:
        logging.error(f'Error in model building: {e}')
        raise e