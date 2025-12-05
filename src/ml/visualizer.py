import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class MLVisualizer:
    """
    Клас для візуалізації результатів машинного навчання.
    Будує порівняльні діаграми метрик та графіки важливості ознак.
    """
    def __init__(self, output_dir="/app/data/visualizations"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({'figure.figsize': (10, 6), 'font.size': 12})

    def plot_model_comparison(self, results_df):
        """Будує графіки порівняння метрик моделей."""
        if results_df.empty:
            return

        # 1. Графік для Регресії (R2 & RMSE)
        reg_df = results_df[results_df["Task"] == "regression"]
        if not reg_df.empty:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Використання hue для уникнення FutureWarning
            sns.barplot(x="Model", y="R2", data=reg_df, ax=axes[0], hue="Model", palette="viridis", legend=False)
            axes[0].set_title("Regression: R2 Score (Higher is better)")
            axes[0].set_ylim(0, 1)

            sns.barplot(x="Model", y="RMSE", data=reg_df, ax=axes[1], hue="Model", palette="magma", legend=False)
            axes[1].set_title("Regression: RMSE (Lower is better)")

            plt.tight_layout()
            save_path = os.path.join(self.output_dir, "regression_comparison.png")
            plt.savefig(save_path)
            print(f"📊 Графік регресії збережено: {save_path}")
            plt.close()

        # 2. Графік для Класифікації
        clf_df = results_df[results_df["Task"] == "classification"]
        if not clf_df.empty:
            metrics_to_plot = ["Accuracy", "F1-Score", "Precision", "Recall"]
            valid_metrics = [m for m in metrics_to_plot if m in clf_df.columns]

            if valid_metrics:
                clf_melted = clf_df.melt(id_vars="Model", value_vars=valid_metrics, var_name="Metric", value_name="Score")

                plt.figure(figsize=(12, 6))
                sns.barplot(x="Model", y="Score", hue="Metric", data=clf_melted, palette="deep")
                plt.title("Classification Metrics Comparison")
                plt.ylim(0, 1)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

                plt.tight_layout()
                save_path = os.path.join(self.output_dir, "classification_comparison.png")
                plt.savefig(save_path)
                print(f"📊 Графік класифікації збережено: {save_path}")
                plt.close()

    def plot_feature_importance(self):
        """Будує графіки важливості ознак на основі збережених CSV файлів."""
        for file in os.listdir(self.output_dir):
            if file.endswith("_importance.csv"):
                df = pd.read_csv(os.path.join(self.output_dir, file))
                model_name = file.replace("_importance.csv", "")

                plt.figure(figsize=(10, 8))
                # Використання hue для уникнення FutureWarning
                sns.barplot(x="Importance", y="Feature", data=df.head(10), hue="Feature", palette="coolwarm", legend=False)
                plt.title(f"Feature Importance: {model_name}")
                plt.tight_layout()

                save_path = os.path.join(self.output_dir, f"{model_name}_plot.png")
                plt.savefig(save_path)
                plt.close()