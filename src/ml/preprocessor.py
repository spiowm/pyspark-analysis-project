import sys
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, expr
from pyspark.ml.feature import VectorAssembler, StandardScaler, Imputer
from pyspark.ml import Pipeline

class MLPreprocessor:
    """
    Клас для попередньої обробки даних перед машинним навчанням.
    Забезпечує очищення, приведення типів та векторизацію.
    """
    def __init__(self, df: DataFrame):
        self.df = df

    def prepare_data(self) -> DataFrame:
        """
        Виконує підготовку даних: фільтрацію, обробку пропусків,
        створення цільових змінних та масштабування ознак.
        """
        print("[Preprocessing] Початок підготовки даних...", flush=True)

        # 1. Фільтрація даних
        # Використовуються лише завершені кредити
        df_ml = self.df.filter(col("loan_status").isin("Fully Paid", "Charged Off", "Default"))

        # Список числових ознак для аналізу
        numeric_cols = [
            "int_rate", "loan_amnt", "term", "installment", "annual_inc", "dti",
            "revol_util", "bc_open_to_buy", "total_bc_limit", "mort_acc",
            "pub_rec", "pub_rec_bankruptcies",
            "delinq_2yrs", "open_acc", "revol_bal", "total_acc",
            "inq_last_6mths", "hardship_flag", "debt_settlement_flag"
        ]

        # Пошук one-hot колонок (створених на етапі ETL)
        home_cols = [c for c in df_ml.columns if "home_ownership" in c and c != "home_ownership"]
        feature_cols = numeric_cols + home_cols

        print(f"[Preprocessing] Кількість вхідних ознак: {len(feature_cols)}", flush=True)

        # 2. Безпечне приведення типів
        print("[Preprocessing] Приведення колонок до типу Double...", flush=True)
        for c in feature_cols:
            if c in df_ml.columns:
                df_ml = df_ml.withColumn(c, expr(f"try_cast(`{c}` as double)"))

        # 3. Обробка пропусків (Imputation)
        valid_cols = [c for c in numeric_cols if c in df_ml.columns]
        print(f"[Preprocessing] Заповнення пропусків середнім значенням ({len(valid_cols)} колонок)...", flush=True)

        imputer = Imputer(inputCols=valid_cols, outputCols=valid_cols).setStrategy("mean")
        try:
            model_imputer = imputer.fit(df_ml)
            df_ml = model_imputer.transform(df_ml)
        except Exception as e:
            print(f"[Preprocessing] Помилка Imputer, використано заповнення нулями. Деталі: {e}")
            df_ml = df_ml.na.fill(0.0, subset=valid_cols)

        # Оновлення списку фінальних колонок після перевірки наявності
        feature_cols = valid_cols + home_cols

        # 4. Обробка цільових змінних (Target)

        # Цільова змінна для регресії (FICO)
        target_reg_col = "fico_range_low"

        # Приведення типу та видалення пропусків у цільовій змінній
        if target_reg_col in df_ml.columns:
            df_ml = df_ml.withColumn(target_reg_col, expr(f"try_cast({target_reg_col} as double)"))
            df_ml = df_ml.na.drop(subset=[target_reg_col])
            df_ml = df_ml.withColumnRenamed(target_reg_col, "label_reg")
            print(f"[Preprocessing] Цільова змінна для регресії встановлена: {target_reg_col}", flush=True)
        else:
            print(f"[Preprocessing] Помилка: Колонка {target_reg_col} відсутня.", flush=True)
            df_ml = df_ml.withColumn("label_reg", col("loan_amnt") * 0.0)

        # Цільова змінна для класифікації (Loan Status)
        df_ml = df_ml.withColumn("label_class",
                                 when(col("loan_status") == "Fully Paid", 0.0).otherwise(1.0)
                                 )

        # 5. Векторизація та Масштабування
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features_raw",
            handleInvalid="skip"
        )

        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=True
        )

        pipeline = Pipeline(stages=[assembler, scaler])

        print("[Preprocessing] Виконання Pipeline (VectorAssembler + StandardScaler)...", flush=True)
        try:
            model = pipeline.fit(df_ml)
            final_df = model.transform(df_ml)
        except Exception as e:
            print(f"[Preprocessing] Критична помилка під час fit/transform: {e}", flush=True)
            raise e

        print("[Preprocessing] Підготовка завершена успішно.", flush=True)
        return final_df.select("features", "label_class", "label_reg")

    def get_feature_names(self):
        """
        Повертає список імен ознак, використаних для навчання.
        Необхідно для побудови графіків важливості ознак.
        """
        df_cols = self.df.columns

        numeric_cols = [
            "int_rate", "loan_amnt", "term", "installment", "annual_inc", "dti",
            "revol_util", "bc_open_to_buy", "total_bc_limit", "mort_acc",
            "pub_rec", "pub_rec_bankruptcies", "delinq_2yrs", "open_acc",
            "revol_bal", "total_acc", "inq_last_6mths", "hardship_flag",
            "debt_settlement_flag"
        ]

        # Фільтрація наявних колонок
        existing_numeric = [c for c in numeric_cols if c in df_cols or c in [x.replace(' ', '_') for x in df_cols]]
        home_cols = [c for c in df_cols if "home_ownership" in c and c != "home_ownership"]

        return existing_numeric + home_cols