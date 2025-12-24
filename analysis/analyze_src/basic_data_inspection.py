from abc import ABC, abstractmethod
import pandas as pd

# Abstract Base Class for Data Inspection Strategies
class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df:pd.DataFrame):
        """Perform a specific type of data inspection"""
        pass

# Concrete Strategy for Data Info Inspection
class DataInfoInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df:pd.DataFrame):
        """Prints the data types and Non-null counts of each columns"""
        print("\nData Types and Non-null Counts:")
        print(df.info())

# Concrete Strategy for Summary Statistics Inspection
class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df:pd.DataFrame):
        """Prints summary statistics for numerical and categorical features"""
        print("\nSummary Statistics (Numerical Features)")
        print(df.describe())
        print("\nSummary Statistics (Categorical Features)")
        print(df.describe(include=["O"]))

# Concrete Strategy for Num of Distinct Values of Columns 
class DistinctColumnsValues(DataInspectionStrategy):
    def inspect(self, df:pd.DataFrame):
        """Prints the distinct values counts of each columns"""
        print("\n:Num of Distinct Columns Values")
        print(df.nunique())

# Implement the Strategy Class that uses DataInspectionStrategy
class DataInspector():
    def __init__(self, strategy:DataInspectionStrategy):
        """Initialize the DataInspector with a specific inspection"""
        self.strategy = strategy
    
    def execute_strategy(self, df:pd.DataFrame):
        """Executes the inspection using the current strategy"""
        self.strategy.inspect(df)