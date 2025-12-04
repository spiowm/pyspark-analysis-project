"""Data loading utilities for the LendingClub dataset."""
import os
import gdown
from pyspark.sql import SparkSession, DataFrame


class DataLoader:
    """Handles data downloading and loading into Spark."""
    
    GOOGLE_DRIVE_FILE_ID = '1sBNFrGdJjDaUdUzlTtiVzzbY_3EY21T1'
    DEFAULT_DATA_PATH = '/app/data/big_data.csv'
    
    def __init__(self, spark: SparkSession, data_path: str = None):
        self.spark = spark
        self.data_path = data_path or self.DEFAULT_DATA_PATH
    
    def download_if_needed(self) -> str:
        """Download dataset from Google Drive if not already present."""
        if not os.path.exists(self.data_path):
            url = f'https://drive.google.com/uc?id={self.GOOGLE_DRIVE_FILE_ID}'
            gdown.download(url, self.data_path, quiet=False)
        return self.data_path
    
    def load(self) -> DataFrame:
        """Download (if needed) and load the dataset into a Spark DataFrame."""
        self.download_if_needed()
        # inferSchema=True is equivalent to low_memory=False in allowing type deduction
        return self.spark.read.csv(self.data_path, header=True, inferSchema=True)
