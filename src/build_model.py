import logging
from abc import ABC, abstractmethod
import pandas as pd

import optuna
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Abstract Base Class for Train Model
class TrainModel(ABC):
    @abstractmethod
    def train(self, df:pd.DataFrame, **kwargs):
        """Abstract method for build and train model"""
        pass
    
    @abstractmethod
    def optimize(self, trial, df:pd.DataFrame):
        """Abstract method to tuning hyper parameters
        
        Args:
            trial: Optuna trial object
            df: data
        """
        pass


class KMeansModel(TrainModel):
    def train(self, df:pd.DataFrame, **kwargs):
        logging.info("Initializing KMeans Model")
        model = KMeans(random_state=42, **kwargs)
        
        predictions = model.fit_predict(df)
        logging.info("KMeans Model training completed")
        return model, predictions
    def optimize(self, trial, df:pd.DataFrame):
        n_clusters = trial.suggest_int("n_clusters", 3, 5)
        init = trial.suggest_categorical("init", ["k-means++", "random"])
        max_iter = trial.suggest_int("max_iter", 100, 1000)
        algorithm = trial.suggest_categorical("algorithm", ["lloyd", "elkan"])
        
        model, predictions = self.train(
            df, init=init, max_iter=max_iter, algorithm=algorithm
        )
        
        score = silhouette_score(df, predictions)
        
        return score

class AgglomerativeModel(TrainModel):
    def train(self, df:pd.DataFrame, **kwargs):
        logging.info("Initializing Agglomerative Clustering Model")
        model = AgglomerativeClustering(**kwargs)
        
        predictions = model.fit_predict(df)
        logging.info("Agglomerative Clustering Model training completed")
        return model, predictions
    def optimize(self, trial, df:pd.DataFrame):
        n_clusters = trial.suggest_int("n_clusters", 3, 5)
        linkage = trial.suggest_categorical("linkage", ["ward", "complete", "average", "single"])
        compute_distances = trial.suggest_categorical("compute_distances", [True, False])
        
        model, predictions = self.train(
            df, n_clusters=n_clusters, linkage=linkage, compute_distances=compute_distances
        )
        
        score = silhouette_score(df, predictions)
        
        return score

class GMModel(TrainModel):
    def train(self, df, **kwargs):
        logging.info("Initializing Gaussian Mixture Model")
        model = GaussianMixture(random_state=42, **kwargs)
        
        predictions = model.fit_predict(df)
        logging.info("Gaussian Mixture Model training completed")
        return model, predictions
    def optimize(self, trial, df:pd.DataFrame):
        n_components = trial.suggest_int("n_components", 3, 5)
        covariance_type = trial.suggest_categorical("covariance_type", ['full', 'tied', 'diag', 'spherical'])
        max_iter = trial.suggest_int("max_iter", 100, 500)
        
        model, predictions = self.train(
            df, n_components=n_components, covariance_type=covariance_type, max_iter=max_iter
        )
        
        # preds = model.predict(df)
        score = silhouette_score(df, predictions)
        
        return score

class HyperparameterTuner:
    """
    Class for performing hyperparameter tuning. It uses Model strategy to perform tuning.
    """
    
    def __init__(self, model, df):
        self.model = model
        self.df = df
    
    def optimize(self, n_trials=100):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.model.optimize(trial, self.df), n_trials=n_trials)
        return study.best_trial.params