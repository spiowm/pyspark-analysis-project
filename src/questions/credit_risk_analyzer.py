from abc import ABC, abstractmethod
from typing import Any, Tuple
from pyspark.sql import DataFrame, Window
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


# ============================================================================
# БІЗНЕС-ПИТАННЯ ПАВЛА
# ============================================================================


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


# ============================================================================
# БІЗНЕС-ПИТАННЯ ОЛЕКСІЯ
# ============================================================================


class StateDefaultRiskAnalysis(BusinessQuestion):
    """
    1. Географічний аналіз ризику дефолту
    Які 10 штатів мають найвищий відсоток charged off кредитів?
    Uses: filters + group by
    """

    question_text = "Топ-10 штатів з найвищим відсотком charged off кредитів:"

    def answer(self, dataframe: DataFrame, **kwargs) -> Tuple[str, Any]:
        # Фільтруємо тільки повністю оплачені та charged off кредити
        filtered_df = dataframe.filter(
            F.col("loan_status").isin(["Fully Paid", "Charged Off"])
        )

        # Групуємо по штату та рахуємо статистику
        state_risk_df = (
            filtered_df.groupBy("addr_state")
            .agg(
                F.count("*").alias("total_loans"),
                F.sum(
                    F.when(F.col("loan_status") == "Charged Off", 1).otherwise(0)
                ).alias("charged_off_count"),
            )
            .withColumn(
                "default_rate_pct",
                (F.col("charged_off_count") / F.col("total_loans") * 100),
            )
            .filter(F.col("total_loans") >= 10)  # Мінімум 10 кредитів для валідності
            .orderBy(F.desc("default_rate_pct"))
            .limit(10)
        )

        return self.question_text, state_risk_df


class CreditHistoryImpactOnRate(BusinessQuestion):
    """
    2. Вплив довжини кредитної історії на процентну ставку
    Як вік кредитної історії корелює з процентною ставкою (аналіз по квартилях)?
    Uses: window function + filters
    """

    question_text = "Вплив віку кредитної історії на процентну ставку (по квартилях):"

    def answer(self, dataframe: DataFrame, **kwargs) -> Tuple[str, Any]:
        # Фільтруємо записи з валідними даними
        filtered_df = dataframe.filter(
            F.col("earliest_cr_line").isNotNull()
            & F.col("issue_d").isNotNull()
            & F.col("int_rate").isNotNull()
        )

        # Конвертуємо дати та рахуємо вік кредитної історії в роках
        # Припускаємо формат "MMM-yyyy" для дат
        df_with_history = filtered_df.withColumn(
            "credit_history_years",
            F.months_between(
                F.to_date(F.col("issue_d"), "MMM-yyyy"),
                F.to_date(F.col("earliest_cr_line"), "MMM-yyyy"),
            )
            / 12,
        ).filter(F.col("credit_history_years") >= 0)

        # Використовуємо window function для розподілу на квартилі
        window_spec = Window.orderBy("credit_history_years")
        df_with_quartiles = df_with_history.withColumn(
            "quartile", F.ntile(4).over(window_spec)
        )

        # Групуємо по квартилях та рахуємо статистику
        quartile_analysis = (
            df_with_quartiles.groupBy("quartile")
            .agg(
                F.min("credit_history_years").alias("min_years"),
                F.max("credit_history_years").alias("max_years"),
                F.avg("int_rate").alias("avg_interest_rate"),
                F.count("*").alias("loan_count"),
            )
            .orderBy("quartile")
        )

        return self.question_text, quartile_analysis


class IndividualVsJointApplications(BusinessQuestion):
    """
    3. Порівняння Individual vs Joint Applications
    Порівняти середню суму кредиту, ставку та відсоток повного погашення
    Uses: filters + group by
    """

    question_text = "Порівняння Individual vs Joint Applications:"

    def answer(self, dataframe: DataFrame, **kwargs) -> Tuple[str, Any]:
        # Фільтруємо записи з валідними application_type
        filtered_df = dataframe.filter(
            F.col("application_type").isin(["Individual", "Joint App"])
        )

        # Групуємо по типу заявки
        comparison_df = (
            filtered_df.groupBy("application_type")
            .agg(
                F.avg("loan_amnt").alias("avg_loan_amount"),
                F.avg("int_rate").alias("avg_interest_rate"),
                F.count("*").alias("total_applications"),
                F.sum(
                    F.when(F.col("loan_status") == "Fully Paid", 1).otherwise(0)
                ).alias("fully_paid_count"),
            )
            .withColumn(
                "full_payment_rate_pct",
                (F.col("fully_paid_count") / F.col("total_applications") * 100),
            )
            .orderBy("application_type")
        )

        return self.question_text, comparison_df


class PurposeDelinquencyAnalysis(BusinessQuestion):
    """
    4. Аналіз мети кредиту та затримок
    Які цілі кредиту найчастіше мають затримки платежів?
    Uses: filters + group by + self-join
    """

    question_text = "Топ-5 цілей кредиту з найвищим рівнем затримок платежів:"

    def answer(self, dataframe: DataFrame, **kwargs) -> Tuple[str, Any]:
        # Фільтруємо записи з валідною метою
        filtered_df = dataframe.filter(F.col("purpose").isNotNull())

        # Створюємо агреговану таблицю по цілях
        purpose_stats = (
            filtered_df.groupBy("purpose")
            .agg(
                F.count("*").alias("total_loans"),
                F.sum(F.when(F.col("delinq_2yrs") > 0, 1).otherwise(0)).alias(
                    "delinquent_loans"
                ),
                F.avg("delinq_2yrs").alias("avg_delinquencies"),
            )
            .alias("stats")
        )

        # Self-join для порівняння з загальною статистикою
        overall_stats = (
            filtered_df.agg(
                F.avg("delinq_2yrs").alias("overall_avg_delinquencies")
            ).alias("overall")
        )

        # Join для отримання відносних показників
        result_df = (
            purpose_stats.crossJoin(overall_stats)
            .withColumn(
                "delinquency_rate_pct",
                (
                    F.col("stats.delinquent_loans")
                    / F.col("stats.total_loans")
                    * 100
                ),
            )
            .withColumn(
                "vs_overall_ratio",
                F.col("stats.avg_delinquencies")
                / F.col("overall.overall_avg_delinquencies"),
            )
            .select(
                F.col("stats.purpose"),
                F.col("stats.total_loans"),
                F.col("stats.delinquent_loans"),
                F.col("delinquency_rate_pct"),
                F.col("stats.avg_delinquencies"),
                F.col("vs_overall_ratio"),
            )
            .orderBy(F.desc("delinquency_rate_pct"))
            .limit(5)
        )

        return self.question_text, result_df


class OpenAccountsImpactAnalysis(BusinessQuestion):
    """
    5. Вплив кількості відкритих рахунків на успішність погашення
    Квартильний аналіз відкритих рахунків vs успішність погашення
    Uses: window function + filters
    """

    question_text = "Вплив кількості відкритих рахунків на успішність погашення (по квартилях):"

    def answer(self, dataframe: DataFrame, **kwargs) -> Tuple[str, Any]:
        # Фільтруємо валідні записи
        filtered_df = dataframe.filter(
            F.col("open_acc").isNotNull()
            & F.col("loan_status").isin(["Fully Paid", "Charged Off"])
        )

        # Використовуємо window function для розподілу на квартилі
        window_spec = Window.orderBy("open_acc")
        df_with_quartiles = filtered_df.withColumn(
            "accounts_quartile", F.ntile(4).over(window_spec)
        )

        # Групуємо по квартилях та аналізуємо успішність
        quartile_analysis = (
            df_with_quartiles.groupBy("accounts_quartile")
            .agg(
                F.min("open_acc").alias("min_accounts"),
                F.max("open_acc").alias("max_accounts"),
                F.avg("open_acc").alias("avg_accounts"),
                F.count("*").alias("total_loans"),
                F.sum(
                    F.when(F.col("loan_status") == "Fully Paid", 1).otherwise(0)
                ).alias("fully_paid_count"),
            )
            .withColumn(
                "success_rate_pct",
                (F.col("fully_paid_count") / F.col("total_loans") * 100),
            )
            .orderBy("accounts_quartile")
        )

        return self.question_text, quartile_analysis


class VerificationStatusImpact(BusinessQuestion):
    """
    6. Аналіз верифікації доходу та успішності
    Порівняння verified vs non-verified клієнтів
    Uses: self-join + filters + group by
    """

    question_text = "Порівняння верифікованих та неверифікованих клієнтів:"

    def answer(self, dataframe: DataFrame, **kwargs) -> Tuple[str, Any]:
        # Фільтруємо валідні записи
        filtered_df = dataframe.filter(
            F.col("verification_status").isNotNull() & F.col("grade").isNotNull()
        )

        # Створюємо агреговану таблицю по статусу верифікації та grade
        verification_stats = (
            filtered_df.groupBy("verification_status", "grade")
            .agg(
                F.count("*").alias("loan_count"),
                F.avg("loan_amnt").alias("avg_loan_amount"),
                F.avg("int_rate").alias("avg_interest_rate"),
                F.sum(
                    F.when(F.col("loan_status") == "Charged Off", 1).otherwise(0)
                ).alias("default_count"),
            )
            .alias("by_verification")
        )

        # Створюємо агреговану таблицю тільки по grade для порівняння
        grade_stats = (
            filtered_df.groupBy("grade")
            .agg(
                F.avg("int_rate").alias("overall_avg_rate"),
                F.count("*").alias("overall_count"),
            )
            .alias("by_grade")
        )

        # Self-join для порівняння
        result_df = (
            verification_stats.join(
                grade_stats, F.col("by_verification.grade") == F.col("by_grade.grade")
            )
            .withColumn(
                "default_rate_pct",
                (
                    F.col("by_verification.default_count")
                    / F.col("by_verification.loan_count")
                    * 100
                ),
            )
            .withColumn(
                "rate_vs_avg",
                F.col("by_verification.avg_interest_rate")
                - F.col("by_grade.overall_avg_rate"),
            )
            .select(
                F.col("by_verification.verification_status"),
                F.col("by_verification.grade"),
                F.col("by_verification.loan_count"),
                F.col("by_verification.avg_loan_amount"),
                F.col("by_verification.avg_interest_rate"),
                F.col("default_rate_pct"),
                F.col("rate_vs_avg"),
            )
            .orderBy("verification_status", "grade")
        )

        return self.question_text, result_df

