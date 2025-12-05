import os
import argparse
from src.dataset.spark_session import create_spark_session
from src.ml.preprocessor import MLPreprocessor
from src.ml.trainer import MLTrainer
from src.ml.visualizer import MLVisualizer

def clean_column_names(df):
    """
    Очищує назви колонок від пробілів та спецсимволів для сумісності зі Spark SQL.
    """
    new_column_names = [
        c.strip().replace(' ', '_').replace('.', '').replace('-', '_').replace('+', '_plus_')
        for c in df.columns
    ]
    return df.toDF(*new_column_names)

def run():
    parser = argparse.ArgumentParser(description="Запуск ML пайплайну.")
    parser.add_argument('--task', type=str, default='all', choices=['classification', 'regression', 'all'])
    parser.add_argument('--model', type=str, default='all', choices=['lr', 'rf', 'gbt', 'all'])
    args = parser.parse_args()

    # 1. Ініціалізація
    spark = create_spark_session(app_name="LendingClubML")
    spark.sparkContext.setLogLevel("ERROR")

    data_path = "/app/data/cleaned_data_merged.csv"
    if not os.path.exists(data_path):
        print(f"Помилка: Файл {data_path} не знайдено.")
        return

    print(f"[Runner] Завантаження даних...")
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .option("multiLine", "true") \
        .option("escape", '"') \
        .csv(data_path)

    # 2. Санітизація та Препроцесинг
    df = clean_column_names(df)

    preprocessor = MLPreprocessor(df)
    ml_data = preprocessor.prepare_data()

    if ml_data is None or ml_data.count() == 0:
        print("Помилка: Дані відсутні після препроцесингу.")
        spark.stop()
        return

    feature_names = preprocessor.get_feature_names()

    # 3. Розбиття на вибірки
    print("[Runner] Розбиття даних на Train / Validation / Test (70/15/15)...")
    train_df, val_df, test_df = ml_data.randomSplit([0.7, 0.15, 0.15], seed=42)

    # 4. Навчання
    trainer = MLTrainer(train_df, val_df, test_df, feature_cols=feature_names)

    if args.task in ['classification', 'all']:
        trainer.run_classification(model_filter=args.model)

    if args.task in ['regression', 'all']:
        trainer.run_regression(model_filter=args.model)

    # 5. Візуалізація
    print("\n[Runner] Генерація графіків та звітів...")
    results_df = trainer.get_results_df()
    visualizer = MLVisualizer()
    visualizer.plot_model_comparison(results_df)
    visualizer.plot_feature_importance()

    print("\n[Runner] ML Пайплайн завершено успішно.")
    spark.stop()

if __name__ == "__main__":
    run()