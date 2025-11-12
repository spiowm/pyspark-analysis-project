from src.questions.credit_risk_analyzer import BusinessQuestion
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from abc import ABC, abstractmethod


class Pipeline(ABC):
    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def _prepare_data(self, df: DataFrame) -> DataFrame:
        pass

    @abstractmethod
    def _load_data(self, file_path: str, file_format: str = "csv") -> DataFrame:
        pass


class QuestionsPipeline(Pipeline):
    def __init__(
        self, steps: list[BusinessQuestion], spark: SparkSession, data_path: str
    ):
        self.steps = steps
        self.spark = spark
        self.df = self._load_data(data_path)

    def _load_data(self, file_path: str, file_format: str = "csv") -> DataFrame:
        """Loads data from a specified path into a Spark DataFrame."""
        print(f"Loading data from {file_path}...")

        # .option("header", "true") to use the first row as column names.
        raw_df = (
            self.spark.read.format(file_format)
            .option("header", "true")
            # --- ВИРІШЕННЯ ПРОБЛЕМИ ---
            # Ця опція дозволяє парсеру коректно обробляти поля,
            # які містять символи переносу рядка (наприклад, довгі описи).
            .option("multiLine", "true")
            # Додаткова корисна опція, яка вказує, що лапки " використовуються
            # для екранування значень, які можуть містити коми.
            .option("escape", '"')
            # .option("inferSchema", "true")
            .load(file_path)
        )

        return self._prepare_data(raw_df)

    def _prepare_data(self, df: DataFrame) -> DataFrame:
        """
        Performs centralized type casting for all numeric columns used in the analysis.
        This is the most robust way to handle data types.
        """
        print("Casting column types...")

        # Define which columns need which types
        cols_to_cast_sql = {
            "loan_amnt": "DOUBLE",
            "annual_inc": "DOUBLE",
            "num_accts_ever_120_pd": "DOUBLE",
            "delinq_2yrs": "DOUBLE",
            "fico_range_low": "DOUBLE",
            "dti": "DOUBLE",
            "total_rec_late_fee": "DOUBLE",
            "tot_hi_cred_lim": "DOUBLE",
            "int_rate": "DOUBLE",
            "open_acc": "DOUBLE",
        }

        prepared_df = df
        for col_name, sql_type in cols_to_cast_sql.items():
            # Use expr to call the SQL TRY_CAST function.
            # This is the most robust way to prevent cast errors.
            prepared_df = prepared_df.withColumn(
                col_name, F.expr(f"TRY_CAST({col_name} AS {sql_type})")
            )

        return prepared_df

    def run(self, name: str):
        print("=" * 60)
        print(f"           ЗАПУСК АНАЛІЗУ БІЗНЕС-ПИТАНЬ {name}")
        print("=" * 60)

        for i, step in enumerate(self.steps):
            question, answer = step.answer(self.df)

            print(f"\n--- Питання {i+1} ---")
            print(f"Питання: {question}")

            if isinstance(answer, DataFrame):
                print("Відповідь:")
                answer.show()
            else:
                print(f"Відповідь: {answer}")

        print("=" * 60)
        print(f"               АНАЛІЗ ПИТАНЬ {name} ЗАВЕРШЕНО")
        print("=" * 60)
