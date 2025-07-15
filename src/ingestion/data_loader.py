import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        logging.info(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path, encoding='latin1') # Common encoding for this dataset
        logging.info("Data loaded successfully.")
        return df
    except FileNotFoundError:
        logging.error(f"File not found at {file_path}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading the data: {e}")
        raise 