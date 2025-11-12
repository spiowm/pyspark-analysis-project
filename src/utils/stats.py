def show_numeric_stats(df):
    """
    Displays summary statistics for numeric columns in a DataFrame.

    Parameters:
    df : The input DataFrame.

    Returns:
    None
    """

    df.describe().show()
