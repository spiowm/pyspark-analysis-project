import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gdown

# PySpark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan, mean, regexp_replace, lit, create_map
from pyspark.sql.types import IntegerType, DoubleType

if __name__=="__main__":
    sns.set()

    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("LendingClubCleaning") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    print("Spark Session created")

    # Download dataset from Google Drive
    import os
    
    # Google Drive file ID
    file_id = '1sBNFrGdJjDaUdUzlTtiVzzbY_3EY21T1'
    url = f'https://drive.google.com/uc?id={file_id}'

    # Save to data/ directory so it persists across container reruns
    output = '/app/data/big_data.csv'
    
    # Only download if not exists to save time during re-runs
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    # Reading CSV file
    # inferSchema=True is equivalent to low_memory=False in allowing type deduction
    raw_data = spark.read.csv(output, header=True, inferSchema=True)

    # Cell 3: Head (Show)
    # Spark doesn't have head() in the same way, show() prints to console
    print("Initial Data Preview:")
    raw_data.show(5)

    # Cell 4: Drop initial columns
    columns_to_drop_initial = ['id', 'member_id', 'grade', 'sub_grade', 'emp_title', 'emp_length']
    raw_data = raw_data.drop(*columns_to_drop_initial)

    print("Data after initial drop:")
    raw_data.show(5)

    # Cell 5: Info equivalent (Schema and Count)
    print(f"Total Rows: {raw_data.count()}")
    print("Schema:")
    raw_data.printSchema()

    # Cell 6: Drop columns with > 30% Nulls
    threshold = 0.3
    total_count = raw_data.count()

    # Calculate missing count for every column efficiently
    # Create an expression for every column to count nulls or NaNs
    null_counts_expr = [
        count(when(col(c).isNull() | (col(c) == 'NaN'), c)).alias(c) 
        for c in raw_data.columns
    ]

    # Execute aggregation
    null_counts = raw_data.select(null_counts_expr).collect()[0].asDict()

    # Identify columns to drop
    cols_to_drop_nulls = [
        k for k, v in null_counts.items() 
        if (v / total_count) > threshold
    ]

    print(f"Dropping {len(cols_to_drop_nulls)} columns due to >30% missing values.")
    raw_data = raw_data.drop(*cols_to_drop_nulls)

    # Cell 7: Check Info after dropping
    print(f"Remaining Columns: {len(raw_data.columns)}")
    # raw_data.printSchema() # Optional to print again

    # Cell 8 & 9: Work with 'loan_amnt'
    # Calculate mean
    mean_loan = raw_data.select(mean(col('loan_amnt'))).collect()[0][0]
    # Fill NA
    raw_data = raw_data.na.fill(mean_loan, ['loan_amnt'])
    # Describe
    raw_data.select("loan_amnt").describe().show()

    # Cell 10, 11, 12: Work with 'funded_amnt'
    mean_funded = raw_data.select(mean(col('funded_amnt'))).collect()[0][0]
    raw_data = raw_data.na.fill(mean_funded, ['funded_amnt'])
    raw_data.select("funded_amnt").describe().show()

    # Cell 13, 14, 15: Work with 'funded_amnt_inv'
    mean_funded_inv = raw_data.select(mean(col('funded_amnt_inv'))).collect()[0][0]
    raw_data = raw_data.na.fill(mean_funded_inv, ['funded_amnt_inv'])
    raw_data.select("funded_amnt_inv").describe().show()

    # Cell 17, 18, 19, 20: Work with 'fico_range_low'
    # Count unique (Approximate is usually faster in Spark, but exact is fine here)
    unique_fico_low = raw_data.select("fico_range_low").distinct().count()
    print(f"Quantity of 'fico_range_low' before drop: {unique_fico_low}")

    # Drop rows where fico_range_low is null
    raw_data = raw_data.na.drop(subset=["fico_range_low"])

    unique_fico_low_after = raw_data.select("fico_range_low").distinct().count()
    print(f"Quantity of 'fico_range_low' after drop: {unique_fico_low_after}")

    # Cell 21, 22, 23, 24: Work with 'fico_range_high'
    unique_fico_high = raw_data.select("fico_range_high").distinct().count()
    print(f"Quantity of 'fico_range_high' before drop: {unique_fico_high}")

    raw_data = raw_data.na.drop(subset=["fico_range_high"])

    unique_fico_high_after = raw_data.select("fico_range_high").distinct().count()
    print(f"Quantity of 'fico_range_high' after drop: {unique_fico_high_after}")

    # Cell 25: Fill all float columns with mean
    # Identify float/double columns
    float_cols = [f.name for f in raw_data.schema.fields if isinstance(f.dataType, DoubleType)]

    # Compute means for all float columns in one pass
    means_dict = raw_data.select([mean(c).alias(c) for c in float_cols]).collect()[0].asDict()

    # Fill NA values using the dictionary of means
    raw_data = raw_data.na.fill(means_dict)

    # Cell 27, 28: Work with 'term'
    # Remove ' months', cast to int, fill nulls with 36
    raw_data = raw_data.withColumn("term", regexp_replace(col("term"), " months", ""))
    raw_data = raw_data.withColumn("term", col("term").cast(IntegerType()))
    raw_data = raw_data.na.fill(36, ["term"])

    print("Term column schema after processing:")
    raw_data.select("term").printSchema()

    # Cell 30, 31, 32, 33: Work with 'home_ownership'
    # Fill NA with 'ANY'
    raw_data = raw_data.na.fill("ANY", ["home_ownership"])

    # Get dummies (One Hot Encoding)
    # Spark doesn't have a direct 1-line get_dummies like Pandas for simple string cols without VectorAssembler steps 
    # usually, but we can pivot or manual loop. Manual loop mimics pandas result best here.
    ownership_categories = [row['home_ownership'] for row in raw_data.select("home_ownership").distinct().collect()]

    for cat in ownership_categories:
        # Create boolean/integer column 1 if match, 0 if not
        col_name = f"home_ownership_{cat}"
        raw_data = raw_data.withColumn(col_name, (col("home_ownership") == cat).cast("integer"))

    # Drop original column
    raw_data = raw_data.drop("home_ownership")

    # Cell 35 - 43: Inspect Object types (just showing unique counts to mimic exploration)
    # Note: collecting distinct values of large columns in Spark can be slow, usually we just count
    object_cols_to_inspect = ['verification_status', 'issue_d', 'loan_status', 'pymnt_plan', 'url', 'purpose', 'title']
    for c in object_cols_to_inspect:
        if c in raw_data.columns:
            # Just printing head of distincts to simulate .unique()
            # raw_data.select(c).distinct().show(5) 
            pass

    # Cell 44: Drop specific object columns
    drop_obj_cols = [
        'verification_status', 'issue_d', 'loan_status', 'pymnt_plan', 'url',
        'zip_code', 'addr_state', 'earliest_cr_line', 'initial_list_status',
        'last_pymnt_d', 'last_credit_pull_d', 'application_type', 'disbursement_method'
    ]
    raw_data = raw_data.drop(*drop_obj_cols)

    # Cell 47, 50, 51: Object data modification
    # Fill 'purpose' and 'title'
    raw_data = raw_data.na.fill("any", ["purpose", "title"])

    # Replace '...' in title with 'any' (assuming '...' string literal)
    raw_data = raw_data.withColumn("title", when(col("title") == "...", "any").otherwise(col("title")))

    # Cell 53 - 56: Work with 'hardship_flag'
    # Map N->0, Y->1, fillna 0
    # Note: Spark's na.fill needs the column type to match. Since it's likely string now 'N'/'Y', we perform logic first.
    raw_data = raw_data.withColumn("hardship_flag", 
        when(col("hardship_flag") == 'Y', 1).otherwise(0)
    )
    # Ensure no nulls (the otherwise 0 covers logic, but strict fill if needed)
    raw_data = raw_data.na.fill(0, ["hardship_flag"])

    # Cell 57 - 60: Work with 'debt_settlement_flag'
    raw_data = raw_data.withColumn("debt_settlement_flag", 
        when(col("debt_settlement_flag") == 'Y', 1).otherwise(0)
    )
    raw_data = raw_data.na.fill(0, ["debt_settlement_flag"])

    # Cell 62, 63: Drop redundant funding columns
    raw_data = raw_data.drop('funded_amnt', 'funded_amnt_inv')

    # Cell 64: Cast home_ownership columns to Int (We already did this during creation step above)
    # But ensuring boolean logic from original notebook
    # The loop in Step 30 already cast them to Integer (0/1). 

    # Cell 73: Visualizations (Histograms)
    # Spark cannot plot directly. We sample data, convert to Pandas, then plot.
    # Taking a sample of 10% or fixed amount for plotting performance
    sample_data = raw_data.select('loan_amnt', 'int_rate', 'installment').sample(fraction=0.1, seed=42).toPandas()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sample_data['loan_amnt'].hist(color='k', bins=30, ax=axes[0])
    axes[0].set_title('Loan Amount')

    sample_data['int_rate'].hist(color='k', bins=30, ax=axes[1])
    axes[1].set_title('Interest Rate')

    sample_data['installment'].hist(color='k', bins=30, ax=axes[2])
    axes[2].set_title('Installment')

    plt.tight_layout()
    plt.show()

    # Cell 74: Visualizations (Boxplots)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sample_data.boxplot(column='loan_amnt', ax=axes[0], color=dict(boxes='k', whiskers='k', medians='r', caps='k'))
    axes[0].set_title('Loan Amount')

    sample_data.boxplot(column='int_rate', ax=axes[1], color=dict(boxes='k', whiskers='k', medians='r', caps='k'))
    axes[1].set_title('Interest Rate')

    sample_data.boxplot(column='installment', ax=axes[2], color=dict(boxes='k', whiskers='k', medians='r', caps='k'))
    axes[2].set_title('Installment')

    plt.tight_layout()
    plt.show()

    # Cell 75: Boxplot specific range
    # Pandas iloc logic raw_data.iloc[:, :18]
    # In spark we just select specific columns.
    subset_cols = raw_data.columns[:18] 
    # Note: boxplotting 18 columns might be messy, but following the structure:
    sample_subset = raw_data.select(subset_cols).sample(fraction=0.1, seed=42).toPandas()
    sample_subset.hist(color="k", bins=30, figsize=(15, 10))
    plt.show()

    # Cell 76: Boxplot first column
    raw_data.select(raw_data.columns[0]).sample(fraction=0.1).toPandas().boxplot(color="k", figsize=(15, 10))
    plt.show()

    # Cell 78: Export
    # PySpark writes to a folder of parts by default. To get a single CSV like pandas:
    # Warning: coalesce(1) moves all data to one node. Only do this if final data fits in memory.
    output_path = '/app/data/cleaned_data_V2.csv'
    print(f"Saving to {output_path}...")

    # Option 1: Standard Spark Write (Folder output)
    raw_data.write.csv(output_path, header=True, mode='overwrite')

    # Option 2: Single CSV file (Pandas style) - Only if data is small enough
    # raw_data.coalesce(1).write.csv(output_path, header=True, mode='overwrite')

    print("Processing Complete.")