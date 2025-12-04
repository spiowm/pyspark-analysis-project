"""
LendingClub Data Processing Pipeline.

This script orchestrates the complete data cleaning and processing pipeline
for the LendingClub dataset using PySpark.
"""
from src.dataset.spark_session import create_spark_session
from src.dataset.loader import DataLoader
from src.dataset.cleaner import DataCleaner
from src.dataset.visualizer import DataVisualizer
from src.dataset.exporter import DataExporter


def main():
    # Initialize Spark Session
    spark = create_spark_session()
    
    # Load data
    loader = DataLoader(spark)
    raw_data = loader.load()
    
    print("Initial Data Preview:")
    raw_data.show(5)
    print(f"Total Rows: {raw_data.count()}")
    print("Schema:")
    raw_data.printSchema()
    
    # Clean data using method chaining
    cleaner = DataCleaner(raw_data)
    cleaned_data = (
        cleaner
        .drop_initial_columns()
        .drop_high_null_columns()
        .fill_numeric_with_mean(['loan_amnt', 'funded_amnt', 'funded_amnt_inv'])
        .drop_null_rows(['fico_range_low', 'fico_range_high'])
        .fill_all_floats_with_mean()
        .process_term_column()
        .one_hot_encode_home_ownership()
        .drop_object_columns()
        .clean_text_columns()
        .encode_flag_columns()
        .drop_redundant_columns()
        .get_dataframe()
    )
    
    # Visualize data
    visualizer = DataVisualizer(cleaned_data)
    visualizer.plot_histograms()
    visualizer.plot_boxplots()
    visualizer.plot_subset_histograms(num_columns=18)
    visualizer.plot_single_column_boxplot(column_index=0)
    
    # Export data
    exporter = DataExporter(cleaned_data)
    exporter.save_as_folder()


if __name__ == "__main__":
    main()
