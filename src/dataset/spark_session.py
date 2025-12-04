"""Spark session management."""
from pyspark.sql import SparkSession


def create_spark_session(
    app_name: str = "LendingClubCleaning",
    driver_memory: str = "4g"
) -> SparkSession:
    """Create and return a configured Spark session."""
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", driver_memory) \
        .getOrCreate()
    print("Spark Session created")
    return spark
