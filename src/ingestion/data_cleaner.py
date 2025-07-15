import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_supply_chain_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the main supply chain DataFrame.

    This function performs several cleaning steps:
    - Converts date columns to datetime objects.
    - Corrects data types for numeric columns.
    - Normalizes categorical text data.
    - Handles missing values.
    - Performs basic feature engineering.

    Args:
        df (pd.DataFrame): The raw DataFrame loaded from the dataset.

    Returns:
        pd.DataFrame: The cleaned and preprocessed DataFrame.
    """
    logging.info("Starting data cleaning process...")
    
    # Create a copy to avoid modifying the original DataFrame
    df_clean = df.copy()

    # Standardize column names (lowercase, replace spaces/special chars with underscores)
    df_clean.columns = (
        df_clean.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )

    # 1. Date and Time Conversion
    # We use 'coerce' to turn any unparseable dates into NaT (Not a Time)
    df_clean['order_date_dateorders'] = pd.to_datetime(df_clean['order_date_dateorders'], errors='coerce')
    df_clean['shipping_date_dateorders'] = pd.to_datetime(df_clean['shipping_date_dateorders'], errors='coerce')

    # 2. Numeric Data Type Correction
    numeric_cols = ['order_item_quantity', 'product_price', 'order_item_total']
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean[col].fillna(0, inplace=True) # Fill non-numeric with 0

    # 3. Text Normalization (lowercase and strip whitespace)
    text_cols = [
        'category_name', 'customer_city', 'customer_country', 'customer_segment',
        'department_name', 'market', 'order_city', 'order_country', 'order_region',
        'order_state', 'order_status', 'product_name', 'shipping_mode', 'delivery_status'
    ]
    for col in text_cols:
        if col in df_clean.columns:
            # Ensure column is of string type before applying string operations
            df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()

    # 4. Handling Missing Values (for key categorical columns)
    # After converting to string and stripping, 'nan' might become a string.
    # We replace these with a more meaningful 'unknown'.
    for col in text_cols:
        df_clean[col].replace('nan', 'unknown', inplace=True)
        df_clean[col].fillna('unknown', inplace=True)

    # 5. Feature Engineering
    # Calculate the number of days it took to ship the order.
    # We handle potential NaT values from the date conversion.
    if 'order_date_dateorders' in df_clean and 'shipping_date_dateorders' in df_clean:
        time_diff = df_clean['shipping_date_dateorders'] - df_clean['order_date_dateorders']
        df_clean['shipping_days'] = time_diff.dt.days
        # Fill missing shipping days with a neutral value like -1 or median if appropriate
        df_clean['shipping_days'].fillna(-1, inplace=True) 

    logging.info("Data cleaning process completed successfully.")
    
    return df_clean 