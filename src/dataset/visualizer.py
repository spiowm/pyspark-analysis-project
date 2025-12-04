"""Visualization utilities for the LendingClub dataset."""
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import DataFrame


class DataVisualizer:
    """Handles all data visualization operations."""
    
    def __init__(self, df: DataFrame, sample_fraction: float = 0.1, seed: int = 42):
        self.df = df
        self.sample_fraction = sample_fraction
        self.seed = seed
        sns.set()
    
    def plot_histograms(self, columns: list = None) -> None:
        """Plot histograms for specified columns."""
        if columns is None:
            columns = ['loan_amnt', 'int_rate', 'installment']
        
        sample_data = self.df.select(columns).sample(
            fraction=self.sample_fraction, seed=self.seed
        ).toPandas()
        
        fig, axes = plt.subplots(1, len(columns), figsize=(15, 5))
        
        for i, column in enumerate(columns):
            sample_data[column].hist(color='k', bins=30, ax=axes[i])
            axes[i].set_title(column.replace('_', ' ').title())
        
        plt.tight_layout()
        plt.show()
    
    def plot_boxplots(self, columns: list = None) -> None:
        """Plot boxplots for specified columns."""
        if columns is None:
            columns = ['loan_amnt', 'int_rate', 'installment']
        
        sample_data = self.df.select(columns).sample(
            fraction=self.sample_fraction, seed=self.seed
        ).toPandas()
        
        fig, axes = plt.subplots(1, len(columns), figsize=(15, 5))
        box_colors = dict(boxes='k', whiskers='k', medians='r', caps='k')
        
        for i, column in enumerate(columns):
            sample_data.boxplot(column=column, ax=axes[i], color=box_colors)
            axes[i].set_title(column.replace('_', ' ').title())
        
        plt.tight_layout()
        plt.show()
    
    def plot_subset_histograms(self, num_columns: int = 18) -> None:
        """Plot histograms for the first N columns."""
        subset_cols = self.df.columns[:num_columns]
        sample_subset = self.df.select(subset_cols).sample(
            fraction=self.sample_fraction, seed=self.seed
        ).toPandas()
        sample_subset.hist(color="k", bins=30, figsize=(15, 10))
        plt.show()
    
    def plot_single_column_boxplot(self, column_index: int = 0) -> None:
        """Plot boxplot for a single column by index."""
        column_name = self.df.columns[column_index]
        self.df.select(column_name).sample(
            fraction=self.sample_fraction
        ).toPandas().boxplot(color="k", figsize=(15, 10))
        plt.show()
