from abc import ABC, abstractmethod
from typing import Any, Tuple
from pyspark.sql import DataFrame, Window, SparkSession
import pyspark.sql.functions as F


class BusinessQuestion(ABC):
    """Абстрактний базовий клас для бізнес-питань."""

    @abstractmethod
    def answer(self, dataframe: DataFrame, **kwargs) -> Tuple[str, Any]:
        """
        Метод для отримання відповіді на бізнес-питання.

        Повертає кортеж, що складається з:
        - Текст питання (str)
        - Відповідь (Any), яка може бути числом, рядком або DataFrame.
        """
        pass


class AvgLoanForHomeowners(BusinessQuestion):
    """
    1. Яка середня сума кредиту для клієнтів із житлом у власності
       та річним доходом понад 100 000?
    """

    question_text = "Яка середня сума кредиту для власників житла з доходом > 100 000?"

    def answer(self, dataframe: DataFrame, **kwargs) -> Tuple[str, Any]:
        result_df = dataframe.filter(
            (F.col("home_ownership") == "OWN") & (F.col("annual_inc") > 100000)
        ).agg(F.avg("loan_amnt").alias("avg_loan_amount"))

        # Витягуємо єдине значення з DataFrame
        first_row = result_df.first()
        answer_value = first_row[0] if first_row else "Дані не знайдені"

        return self.question_text, f"${answer_value:,.2f}" if isinstance(
            answer_value, (int, float)
        ) else answer_value


class HighRiskDebtorsCount(BusinessQuestion):
    """
    2. Скільки клієнтів мають прострочення понад 120 днів і зверталися
       за кредитом на покриття боргів?
    """

    question_text = "Скільки клієнтів з простроченням > 120 днів брали кредит для консолідації боргу?"

    def answer(self, dataframe: DataFrame, **kwargs) -> Tuple[str, Any]:
        # .count() одразу повертає число, що ідеально підходить
        answer_value = dataframe.filter(
            (F.col("num_accts_ever_120_pd") > 0)
            & (F.col("purpose") == "debt_consolidation")
        ).count()
        return self.question_text, answer_value


class LowFicoDelinquencyRate(BusinessQuestion):
    """
    3. Який відсоток клієнтів з рейтингом FICO нижче 650
       мають прострочені платежі за останні 2 роки?
    """

    question_text = (
        "Який відсоток клієнтів з FICO < 650 мають прострочення за останні 2 роки?"
    )

    def answer(self, dataframe: DataFrame, **kwargs) -> Tuple[str, Any]:
        low_fico_df = dataframe.filter(F.col("fico_range_low") < 650)
        total_count = low_fico_df.count()

        if total_count == 0:
            return self.question_text, "Клієнти з FICO < 650 не знайдені."

        delinquent_count = low_fico_df.filter(F.col("delinq_2yrs") > 0).count()
        delinquency_rate = (delinquent_count / total_count) * 100

        return self.question_text, f"{delinquency_rate:.2f}%"


class StateCreditProfile(BusinessQuestion):
    """
    4. Для кожного штату розрахуйте середній DTI та загальну суму
       штрафів за прострочення.
    """

    question_text = (
        "Середній DTI та загальна сума штрафів за прострочення по кожному штату:"
    )

    def answer(self, dataframe: DataFrame, **kwargs) -> Tuple[str, Any]:
        # Для цього питання відповідь - це таблиця (DataFrame)
        state_agg_df = (
            dataframe.groupBy("addr_state")
            .agg(
                F.avg("dti").alias("avg_dti"),
                F.sum("total_rec_late_fee").alias("total_late_fees"),
            )
            .orderBy(F.desc("total_late_fees"))
        )

        return self.question_text, state_agg_df


class ProfessionCreditLimit(BusinessQuestion):
    """
    5. Для кожної професії визначте середній кредитний ліміт та
       кількість клієнтів (топ-10).
    """

    question_text = (
        "Топ-10 професій за кількістю клієнтів та їх середній кредитний ліміт:"
    )

    def answer(self, dataframe: DataFrame, **kwargs) -> Tuple[str, Any]:
        # Відповідь - це також DataFrame
        profession_agg_df = (
            dataframe.filter(F.col("emp_title").isNotNull())
            .groupBy("emp_title")
            .agg(
                F.avg("tot_hi_cred_lim").alias("avg_credit_limit"),
                F.count("*").alias("num_clients"),
            )
            .orderBy(F.desc("num_clients"))
            .limit(10)
        )

        return self.question_text, profession_agg_df


class IncomeDifferenceByState(BusinessQuestion):
    """
    6. Розрахунок різниці між річним доходом клієнта та середнім доходом у його штаті.
    """

    question_text = "Різниця між доходом клієнта та середнім доходом по його штату (випадковий приклад 10 рядків):"

    def answer(self, dataframe: DataFrame, **kwargs) -> Tuple[str, Any]:
        window_spec = Window.partitionBy("addr_state")
        avg_income_by_state = F.avg("annual_inc").over(window_spec)

        result_df = (
            dataframe.withColumn("avg_state_income", avg_income_by_state)
            .withColumn(
                "income_difference", F.col("annual_inc") - F.col("avg_state_income")
            )
            .orderBy(F.rand())
            .select(
                "id",
                "addr_state",
                "annual_inc",
                "avg_state_income",
                "income_difference",
            )
            .limit(10)
        )

        return self.question_text, result_df
