"""Data cleaning operations for the LendingClub dataset."""
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, when, mean, regexp_replace
from pyspark.sql.types import IntegerType, DoubleType


class DataCleaner:
    """Handles all data cleaning operations."""
    
    # Columns to drop at the start
    INITIAL_DROP_COLUMNS = ['id', 'member_id', 'grade', 'sub_grade', 'emp_title', 'emp_length']
    
    # Object columns to drop
    OBJECT_COLUMNS_TO_DROP = [
        'verification_status', 'issue_d', 'pymnt_plan', 'url',
        'zip_code', 'addr_state', 'earliest_cr_line', 'initial_list_status',
        'last_pymnt_d', 'last_credit_pull_d', 'application_type', 'disbursement_method'
    ]
    
    # Columns to drop at the end (redundant)
    REDUNDANT_COLUMNS = ['funded_amnt', 'funded_amnt_inv']
    
    # Threshold for dropping columns with too many nulls
    NULL_THRESHOLD = 0.3
    
    def __init__(self, df: DataFrame):
        self.df = df
    
    def drop_initial_columns(self) -> 'DataCleaner':
        """Drop initial unnecessary columns."""
        self.df = self.df.drop(*self.INITIAL_DROP_COLUMNS)
        print("Data after initial drop:")
        self.df.show(5)
        return self
    
    def drop_high_null_columns(self) -> 'DataCleaner':
        """Drop columns with more than 30% null values."""
        total_count = self.df.count()
        
        # Calculate missing count for every column efficiently
        null_counts_expr = [
            count(when(col(c).isNull() | (col(c) == 'NaN'), c)).alias(c) 
            for c in self.df.columns
        ]
        
        null_counts = self.df.select(null_counts_expr).collect()[0].asDict()
        
        # Identify columns to drop
        cols_to_drop = [
            k for k, v in null_counts.items() 
            if (v / total_count) > self.NULL_THRESHOLD
        ]
        
        print(f"Dropping {len(cols_to_drop)} columns due to >30% missing values.")
        self.df = self.df.drop(*cols_to_drop)
        print(f"Remaining Columns: {len(self.df.columns)}")
        return self
    
    def fill_numeric_with_mean(self, columns: list) -> 'DataCleaner':
        """Fill NA values in specified columns with their mean."""
        for column in columns:
            mean_val = self.df.select(mean(col(column))).collect()[0][0]
            self.df = self.df.na.fill(mean_val, [column])
            self.df.select(column).describe().show()
        return self
    
    def drop_null_rows(self, columns: list) -> 'DataCleaner':
        """Drop rows where specified columns have null values."""
        for column in columns:
            unique_before = self.df.select(column).distinct().count()
            print(f"Quantity of '{column}' before drop: {unique_before}")
            
            self.df = self.df.na.drop(subset=[column])
            
            unique_after = self.df.select(column).distinct().count()
            print(f"Quantity of '{column}' after drop: {unique_after}")
        return self
    
    def fill_all_floats_with_mean(self) -> 'DataCleaner':
        """Fill all float/double columns with their mean values."""
        float_cols = [f.name for f in self.df.schema.fields if isinstance(f.dataType, DoubleType)]
        means_dict = self.df.select([mean(c).alias(c) for c in float_cols]).collect()[0].asDict()
        self.df = self.df.na.fill(means_dict)
        return self
    
    def process_term_column(self) -> 'DataCleaner':
        """Clean the term column: remove ' months', cast to int, fill nulls with 36."""
        self.df = self.df.withColumn("term", regexp_replace(col("term"), " months", ""))
        self.df = self.df.withColumn("term", col("term").cast(IntegerType()))
        self.df = self.df.na.fill(36, ["term"])
        print("Term column schema after processing:")
        self.df.select("term").printSchema()
        return self
    
    def one_hot_encode_home_ownership(self) -> 'DataCleaner':
        """One-hot encode the home_ownership column."""
        self.df = self.df.na.fill("ANY", ["home_ownership"])
        
        ownership_categories = [
            row['home_ownership'] 
            for row in self.df.select("home_ownership").distinct().collect()
        ]
        
        for cat in ownership_categories:
            col_name = f"home_ownership_{cat}"
            self.df = self.df.withColumn(col_name, (col("home_ownership") == cat).cast("integer"))
        
        self.df = self.df.drop("home_ownership")
        return self
    
    def drop_object_columns(self) -> 'DataCleaner':
        """Drop specified object/string columns."""
        self.df = self.df.drop(*self.OBJECT_COLUMNS_TO_DROP)
        return self
    
    def clean_text_columns(self) -> 'DataCleaner':
        """Clean purpose and title columns."""
        self.df = self.df.na.fill("any", ["purpose", "title"])
        self.df = self.df.withColumn("title", when(col("title") == "...", "any").otherwise(col("title")))
        return self
    
    def encode_flag_columns(self) -> 'DataCleaner':
        """Convert Y/N flag columns to 1/0."""
        for flag_col in ["hardship_flag", "debt_settlement_flag"]:
            self.df = self.df.withColumn(
                flag_col, 
                when(col(flag_col) == 'Y', 1).otherwise(0)
            )
            self.df = self.df.na.fill(0, [flag_col])
        return self
    
    def drop_redundant_columns(self) -> 'DataCleaner':
        """Drop redundant funding columns."""
        self.df = self.df.drop(*self.REDUNDANT_COLUMNS)
        return self
    
    def get_dataframe(self) -> DataFrame:
        """Return the cleaned DataFrame."""
        return self.df
