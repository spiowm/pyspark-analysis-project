"""Data export utilities for the LendingClub dataset."""
from pyspark.sql import DataFrame


class DataExporter:
    """Handles data export operations."""
    
    DEFAULT_OUTPUT_PATH = '/app/data/cleaned-data'
    
    def __init__(self, df: DataFrame, output_path: str = None):
        self.df = df
        self.output_path = output_path or self.DEFAULT_OUTPUT_PATH
    
    def save_as_folder(self) -> None:
        """Save as Spark's default folder of part files."""
        print(f"Saving to {self.output_path}...")
        self.df.write.csv(self.output_path, header=True, mode='overwrite')
        print("Processing Complete.")
    
    def save_as_single_file(self) -> None:
        """
        Save as a single CSV file.
        Warning: coalesce(1) moves all data to one node. 
        Only do this if final data fits in memory.
        """
        print(f"Saving to {self.output_path}...")
        self.df.coalesce(1).write.csv(self.output_path, header=True, mode='overwrite')
        print("Processing Complete.")
