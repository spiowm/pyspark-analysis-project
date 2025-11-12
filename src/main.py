from pyspark.sql import SparkSession
import pandas as pd

from src.utils.stats import show_numeric_stats
from src.questions.pipeline import QuestionsPipeline
from src.questions.credit_risk_analyzer import (
    AvgLoanForHomeowners,
    HighRiskDebtorsCount,
    LowFicoDelinquencyRate,
    StateCreditProfile,
    ProfessionCreditLimit,
    IncomeDifferenceByState,
)


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
    show_numeric_stats(spark, "data/accepted_credit_scores.csv")

    # Бізнес-питання від Павла
    pipeline = QuestionsPipeline(
        steps=[
            AvgLoanForHomeowners(),
            HighRiskDebtorsCount(),
            LowFicoDelinquencyRate(),
            StateCreditProfile(),
            ProfessionCreditLimit(),
            IncomeDifferenceByState(),
        ],
        spark=spark,
        data_path="data/accepted_credit_scores.csv",
    )
    pipeline.run("ПАВЛА")

    spark.stop()


if __name__ == "__main__":
    main()
