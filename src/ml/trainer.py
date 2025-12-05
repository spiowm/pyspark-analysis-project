import time
import os
import pandas as pd
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator

class MLTrainer:
    """
    Клас для навчання, оцінки та збереження моделей машинного навчання.
    """
    def __init__(self, train_df, val_df, test_df, feature_cols, models_save_dir="/app/data/models"):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.feature_cols = feature_cols
        self.save_dir = models_save_dir
        self.results_history = []
        os.makedirs(self.save_dir, exist_ok=True)

    def _train_and_evaluate(self, name, model, evaluator_dict, task_type):
        """
        Навчає модель, виконує оцінку на трьох вибірках (Train/Val/Test) та зберігає артефакти.
        """
        print(f"\n>>> Початок навчання: {name}")
        start_time = time.time()

        # Навчання
        fitted_model = model.fit(self.train_df)
        duration = time.time() - start_time
        print(f">>> Навчання завершено. Час: {duration:.2f} сек.")

        # Розрахунок метрик
        print("   >>> Оцінка моделі (Train / Validation / Test)...")
        pred_train = fitted_model.transform(self.train_df)
        pred_val = fitted_model.transform(self.val_df)
        pred_test = fitted_model.transform(self.test_df)

        metrics_record = {"Model": name, "Task": task_type, "Duration": duration}

        datasets = [("TRAIN", pred_train), ("VALIDATION", pred_val), ("TEST", pred_test)]

        for stage, preds in datasets:
            print(f"   [{stage}]")
            for metric_name, evaluator in evaluator_dict.items():
                metric_val = evaluator.evaluate(preds)
                print(f"      {metric_name}: {metric_val:.4f}")

                # Збереження метрик тестової вибірки для звіту
                if stage == "TEST":
                    metrics_record[metric_name] = metric_val

        self.results_history.append(metrics_record)

        # Аналіз важливості ознак
        self._extract_feature_importance(fitted_model, name, task_type)

        # Збереження моделі
        model_path = os.path.join(self.save_dir, f"{task_type}_{name.replace(' ', '_')}")
        try:
            fitted_model.write().overwrite().save(model_path)
            print(f"   >>> Модель збережено в: {model_path}")
        except Exception:
            pass

        print("-" * 60)

    def _extract_feature_importance(self, model, model_name, task_type):
        """Витягує та зберігає важливість ознак у CSV."""
        importances = None
        if hasattr(model, "featureImportances"):
            importances = model.featureImportances.toArray()
        elif hasattr(model, "coefficients"):
            importances = [abs(x) for x in model.coefficients.toArray()]

        if importances is not None and len(importances) == len(self.feature_cols):
            fi_df = pd.DataFrame({
                "Feature": self.feature_cols,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(10)

            csv_path = f"/app/data/visualizations/{task_type}_{model_name.replace(' ', '_')}_importance.csv"
            os.makedirs("/app/data/visualizations", exist_ok=True)
            fi_df.to_csv(csv_path, index=False)

            # Вивід топ-3 для контролю
            top_3 = fi_df['Feature'].head(3).tolist()
            print(f"   >>> Топ-3 важливі ознаки: {', '.join(top_3)}")

    def run_classification(self, model_filter="all"):
        print("\n" + "="*60)
        print("ЗАДАЧА 1: КЛАСИФІКАЦІЯ (Передбачення дефолту)")
        print("="*60)

        models = {
            "lr": ("Logistic Regression", LogisticRegression(featuresCol="features", labelCol="label_class", maxIter=10)),
            "rf": ("Random Forest", RandomForestClassifier(featuresCol="features", labelCol="label_class", numTrees=30, maxDepth=10)),
            "gbt": ("GBT Classifier", GBTClassifier(featuresCol="features", labelCol="label_class", maxIter=20, maxDepth=8))
        }

        evaluators = {
            "Accuracy": MulticlassClassificationEvaluator(labelCol="label_class", metricName="accuracy"),
            "F1-Score": MulticlassClassificationEvaluator(labelCol="label_class", metricName="f1"),
            "Precision": MulticlassClassificationEvaluator(labelCol="label_class", metricName="weightedPrecision"),
            "Recall": MulticlassClassificationEvaluator(labelCol="label_class", metricName="weightedRecall")
        }

        self._execute_models(models, model_filter, evaluators, "classification")

    def run_regression(self, model_filter="all"):
        print("\n" + "="*60)
        print("ЗАДАЧА 2: РЕГРЕСІЯ (Передбачення FICO Score)")
        print("="*60)

        models = {
            "lr": ("Linear Regression", LinearRegression(featuresCol="features", labelCol="label_reg")),

            "rf": ("Random Forest", RandomForestRegressor(
                featuresCol="features",
                labelCol="label_reg",
                numTrees=50,
                maxDepth=10,
                seed=42
            )),

            "gbt": ("GBT Regressor", GBTRegressor(
                featuresCol="features",
                labelCol="label_reg",
                maxIter=70,
                maxDepth=7,
                stepSize=0.1,
                subsamplingRate=0.7,
                seed=42
            ))
        }

        evaluators = {
            "RMSE": RegressionEvaluator(labelCol="label_reg", metricName="rmse"),
            "R2": RegressionEvaluator(labelCol="label_reg", metricName="r2")
        }

        self._execute_models(models, model_filter, evaluators, "regression")

    def _execute_models(self, models_map, filter_key, evaluators, task_name):
        if filter_key == "all":
            selected_models = models_map.values()
        elif filter_key in models_map:
            selected_models = [models_map[filter_key]]
        else:
            print(f"Модель '{filter_key}' не знайдена.")
            return

        for pretty_name, model_instance in selected_models:
            self._train_and_evaluate(pretty_name, model_instance, evaluators, task_name)

    def get_results_df(self):
        return pd.DataFrame(self.results_history)