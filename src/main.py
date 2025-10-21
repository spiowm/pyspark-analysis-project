from pyspark.sql import SparkSession
import pandas as pd


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

    spark.stop()


if __name__ == "__main__":
    main()
