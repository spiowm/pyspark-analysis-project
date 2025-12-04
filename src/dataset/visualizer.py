"""Visualization utilities for the LendingClub dataset."""
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import DataFrame


class DataVisualizer:
    """Handles all data visualization operations."""
    
    DEFAULT_OUTPUT_DIR = '/app/data/visualizations'
    
    def __init__(self, df: DataFrame, sample_fraction: float = 0.1, seed: int = 42, output_dir: str = None):
        self.df = df
        self.sample_fraction = sample_fraction
        self.seed = seed
        self.output_dir = output_dir or self.DEFAULT_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set()
    
    def _save_figure(self, filename: str) -> None:
        """Save the current figure to the output directory."""
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
    
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
        self._save_figure('histograms.png')
    
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
        self._save_figure('boxplots.png')
    
    def plot_subset_histograms(self, num_columns: int = 18) -> None:
        """Plot histograms for the first N columns."""
        subset_cols = self.df.columns[:num_columns]
        sample_subset = self.df.select(subset_cols).sample(
            fraction=self.sample_fraction, seed=self.seed
        ).toPandas()
        sample_subset.hist(color="k", bins=30, figsize=(15, 10))
        self._save_figure('subset_histograms.png')
    
    def plot_single_column_boxplot(self, column_index: int = 0) -> None:
        """Plot boxplot for a single column by index."""
        column_name = self.df.columns[column_index]
        self.df.select(column_name).sample(
            fraction=self.sample_fraction
        ).toPandas().boxplot(color="k", figsize=(15, 10))
        self._save_figure('single_column_boxplot.png')
