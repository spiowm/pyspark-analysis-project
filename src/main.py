from pyspark.sql import SparkSession
import pandas as pd

from src.utils.stats import show_numeric_stats


def main():
    spark = (
        SparkSession.builder.appName("PySpark Analysis Project")
        .master("local[*]")
        .getOrCreate()
    )

    print("=" * 60)
    print("Spark Session created successfully using the Official Docker Image!")
    print(f"Python Version: {spark.sparkContext.pythonVer}")
    print(f"Spark Version: {spark.version}")

    # Перевіряємо, чи доступний pandas
    pd_version = pd.__version__
    print(f"Pandas Version: {pd_version}")
    print("=" * 60)

    # Створити тестовий датафрейм
    df_we = spark.createDataFrame(
        data=[("Pavlo", 20), ("Oleksiy", 20)], schema=["name", "age"]
    )
    df_we.show()

    # Код для аналізу даних буде тут
    df = spark.read.csv("data/accepted_credit_scores.csv")
    show_numeric_stats(df)

    spark.stop()


if __name__ == "__main__":
    main()
