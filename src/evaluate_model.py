import logging
from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

# Abstract Base Class for Model Evaluation Strategy
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(self, df: pd.DataFrame, labels):
        """
        Abstract method to evaluate a model.
        
        Args:
        model: The trained model to evaluate.
        df (pd.DataFrame): The data features.
        """
        pass


# Concrete Class for Internal Evaluation
class InternalEvaluation(ModelEvaluationStrategy):
    def evaluate_model(self, df: pd.DataFrame, labels):
        """Evaluates model using internal evaluation metrics"""
        score1 = silhouette_score(df, labels)
        score2 = davies_bouldin_score(df, labels)
        score3 = calinski_harabasz_score(df, labels)
        
        logging.info(f'Silhouette Score: {score1:.4f}')
        logging.info(f'Davies Bouldin Score: {score2:.4f}')
        logging.info(f'Calinski Harabasz Score: {score3:.4f}')
        return score1, score2, score3

# Concrete Class for Visual Evaluation
class VisualEvaluation(ModelEvaluationStrategy):
    def evaluate_model(self, df:pd.DataFrame, labels):
        """Evaluates model using `PCA` for dimensionality reduction and `matplotlib` for visualization"""
        pca = PCA(n_components=2)
        df_pca = pca.fit_transform(df)    
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('Customer Segments Visualization')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.show()

# Concrete Class for Business Evaluation
class BusinessEvaluation(ModelEvaluationStrategy):
    def evaluate_model(self, df:pd.DataFrame, labels):
        """Evaluates model using business evaluation metrics"""
        df['Segment'] = labels
        
        segment_analysis = df.groupby('Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'CustomerID': 'count'
        }).round(2)
        
        segment_analysis.columns = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Count']
        
        print("\n=== Segment Analysis ===")
        print(segment_analysis)
        
        return segment_analysis